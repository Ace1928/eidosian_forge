from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.util import resource as resource_lib  # pylint: disable=unused-import
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.concepts import resource_parameter_info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
class CompleterInfo(object):
    """Holds data that can be used to instantiate a resource completer."""

    def __init__(self, static_params=None, id_field=None, collection_info=None, method=None, param_name=None):
        self.static_params = static_params
        self.id_field = id_field
        self.collection_info = collection_info
        self.method = method
        self.param_name = param_name

    @classmethod
    def FromResource(cls, resource_spec, attribute_name):
        """Gets the method, param_name, and other configuration for a completer.

    Args:
      resource_spec: concepts.ResourceSpec, the overall resource.
      attribute_name: str, the name of the attribute whose argument will use
        this completer.

    Raises:
      AttributeError: if the attribute doesn't belong to the resource.

    Returns:
      CompleterInfo, the instantiated object.
    """
        for a in resource_spec.attributes:
            if a.name == attribute_name:
                attribute = a
                break
        else:
            raise AttributeError('Attribute [{}] not found in resource.'.format(attribute_name))
        param_name = resource_spec.ParamName(attribute_name)
        static_params = attribute.completion_request_params
        id_field = attribute.completion_id_field
        collection_info = _GetCompleterCollectionInfo(resource_spec, attribute)
        if collection_info.full_name in _SPECIAL_COLLECTIONS_MAP:
            special_info = _SPECIAL_COLLECTIONS_MAP.get(collection_info.full_name)
            method = registry.GetMethod(collection_info.full_name, 'list')
            static_params = special_info.static_params
            id_field = special_info.id_field
            param_name = special_info.param_name
        if not collection_info:
            return CompleterInfo(static_params, id_field, None, None, param_name)
        try:
            method = registry.GetMethod(collection_info.full_name, 'list', api_version=collection_info.api_version)
        except registry.UnknownMethodError:
            if collection_info.full_name != _PROJECTS_COLLECTION and collection_info.full_name.split('.')[-1] == 'projects':
                special_info = _SPECIAL_COLLECTIONS_MAP.get(_PROJECTS_COLLECTION)
                method = registry.GetMethod(_PROJECTS_COLLECTION, 'list')
                static_params = special_info.static_params
                id_field = special_info.id_field
            else:
                method = None
        except registry.Error:
            method = None
        return CompleterInfo(static_params, id_field, collection_info, method, param_name)

    def GetMethod(self):
        """Get the APIMethod for an attribute in a resource."""
        return self.method