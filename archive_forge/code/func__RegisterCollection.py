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
def _RegisterCollection(self, collection_info):
    """Registers given collection with registry.

    Args:
      collection_info: CollectionInfo, description of resource collection.
    Raises:
      AmbiguousAPIException: If the API defines a collection that has already
          been added.
      AmbiguousResourcePath: If api uses same path for multiple resources.
    """
    api_name = collection_info.api_name
    api_version = collection_info.api_version
    parser = _ResourceParser(self, collection_info)
    collection_parsers = self.parsers_by_collection.setdefault(api_name, {}).setdefault(api_version, {})
    collection_subpaths = collection_info.flat_paths
    if not collection_subpaths:
        collection_subpaths = {'': collection_info.path}
    for subname, path in six.iteritems(collection_subpaths):
        collection_name = collection_info.full_name + ('.' + subname if subname else '')
        existing_parser = collection_parsers.get(collection_name)
        if existing_parser is not None:
            raise AmbiguousAPIException(collection_name, [collection_info.base_url, existing_parser.collection_info.base_url])
        collection_parsers[collection_name] = parser
        if collection_info.enable_uri_parsing:
            self._AddParserForUriPath(api_name, api_version, subname, parser, path)