from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
class _ResourceParser(object):
    """Class that turns command-line arguments into a cloud resource message."""

    def __init__(self, registry, collection_info):
        """Create a _ResourceParser for a given collection.

    Args:
      registry: Registry, The resource registry this parser belongs to.
      collection_info: resource_util.CollectionInfo, description for collection.
    """
        self.registry = registry
        self.collection_info = collection_info

    def ParseRelativeName(self, relative_name, base_url=None, subcollection='', url_unescape=False):
        """Parse relative name into a Resource object.

    Args:
      relative_name: str, resource relative name.
      base_url: str, base url part of the api which manages this resource.
      subcollection: str, id of subcollection. See the api resource module
          (googlecloudsdk/generated_clients/apis/API_NAME/API_VERSION/resources.py).
      url_unescape: bool, if true relative name parameters will be unescaped.

    Returns:
      Resource representing this name.

    Raises:
      InvalidResourceException: if relative name doesn't match collection
          template.
    """
        base_url = apis_internal.UniversifyAddress(base_url)
        path_template = self.collection_info.GetPathRegEx(subcollection)
        match = re.match(path_template, relative_name)
        if not match:
            raise InvalidResourceException(relative_name, 'It is not in {0} collection as it does not match path template {1}'.format(self.collection_info.full_name, path_template))
        params = self.collection_info.GetParams(subcollection)
        fields = match.groups()
        if url_unescape:
            fields = map(urllib.parse.unquote, fields)
        return Resource(self.registry, self.collection_info, subcollection, param_values=dict(zip(params, fields)), endpoint_url=base_url)

    def ParseResourceId(self, resource_id, kwargs, base_url=None, subcollection='', validate=True, default_resolver=None):
        """Given a command line and some keyword args, get the resource.

    Args:
      resource_id: str, Some identifier for the resource.
          Can be None to indicate all params should be taken from kwargs.
      kwargs: {str:(str or func()->str)}, flags (available from context) or
          resolvers that can help parse this resource. If the fields in
          collection-path do not provide all the necessary information,
          kwargs will be searched for what remains.
      base_url: use this base url (endpoint) for the resource, if not provided
          default corresponding api version base url will be used.
      subcollection: str, name of subcollection to use when parsing this path.
      validate: bool, Validate syntax. Use validate=False to handle IDs under
        construction. An ID can be:
          fully qualified - All parameters are specified and have valid syntax.
          partially qualified - Some parameters are specified, all have valid
            syntax.
          under construction - Some parameters may be missing or too short and
            not meet the syntax constraints. With additional characters they
            would have valid syntax. Used by completers that build IDs from
            strings character by character. Completers need to do the
            string => parameters => string round trip with validate=False to
            handle the "add character TAB" cycle.
      default_resolver: func(str) => str, a default param resolver function
        called if kwargs doesn't resolve a param.

    Returns:
      protorpc.messages.Message, The object containing info about this resource.

    Raises:
      InvalidResourceException: If the provided collection-path is malformed.
      WrongResourceCollectionException: If the collection-path specified the
          wrong collection.
      RequiredFieldOmittedException: If the collection-path's path did not
          provide enough fields.
      GRIPathMismatchException: If the number of path segments in the GRI does
          not match the expected format of the URL for the given resource
          collection.
      ValueError: if parameter set in kwargs is not subset of the resource
          parameters.
    """
        base_url = apis_internal.UniversifyAddress(base_url)
        if resource_id is not None:
            try:
                return self.ParseRelativeName(resource_id, base_url=base_url, subcollection=subcollection)
            except InvalidResourceException:
                path = self.collection_info.GetPath(subcollection)
                path_prefixes = self.GetFieldNamesFromPath(path)
                contains_all_fields = all((prefix + '/' in resource_id for prefix in path_prefixes))
                if contains_all_fields:
                    raise UserError('Invalid value: {}'.format(resource_id))
                else:
                    pass
        params = self.collection_info.GetParams(subcollection)
        if not set(kwargs.keys()).issubset(params):
            raise ValueError('Provided params {} is not subset of the resource parameters {} for collection {}'.format(sorted(kwargs.keys()), sorted(params), self.collection_info.full_name))
        if _GRIsAreEnabled():
            gri = GRI.FromString(resource_id, collection=self.collection_info.full_name, validate=validate)
            fields = gri.path_fields
            if len(fields) > len(params):
                raise GRIPathMismatchException(resource_id, params, collection=gri.collection if gri.is_fully_qualified else None)
            elif len(fields) < len(params):
                fields += [None] * (len(params) - len(fields))
            fields = reversed(fields)
        else:
            fields = [None] * len(params)
            fields[-1] = resource_id
        param_values = dict(zip(params, fields))
        for param, value in param_values.items():
            if value is not None:
                continue
            resolver = kwargs.get(param)
            if resolver:
                param_values[param] = resolver() if callable(resolver) else resolver
            elif default_resolver:
                param_values[param] = default_resolver(param)
        ref = Resource(self.registry, self.collection_info, subcollection, param_values, base_url)
        return ref

    def GetFieldNamesFromPath(self, path):
        """Extract field names from uri template path.

    Args:
      path: str, uri template path.

    Returns:
      list(str), list of field names in the template path.
    """
        return [prefix for idx, prefix in enumerate(path.split('/')) if idx % 2 == 0 and prefix]

    def __str__(self):
        path_str = ''
        for param in self.collection_info.params:
            path_str = '[{path}]/{param}'.format(path=path_str, param=param)
        return '[{collection}::]{path}'.format(collection=self.collection_info.full_name, path=path_str)