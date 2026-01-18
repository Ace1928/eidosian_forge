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
class ParentTranslator(object):
    """Translates parent collections for completers.

  Attributes:
    collection: str, the collection name.
    param_translation: {str: str}, lookup from the params of the child
      collection to the params of the special parent collection. If None,
      then the collections match and translate methods are a no-op.
  """

    def __init__(self, collection, param_translation=None):
        self.collection = collection
        self.param_translation = param_translation or {}

    def ToChildParams(self, params):
        """Translate from original parent params to params that match the child."""
        if self.param_translation:
            for orig_param, new_param in six.iteritems(self.param_translation):
                params[orig_param] = params.get(new_param)
                del params[new_param]
        return params

    def MessageResourceMap(self, message, ref):
        """Get dict for translating parent params into the given message type."""
        message_resource_map = {}
        for orig_param, special_param in six.iteritems(self.param_translation):
            try:
                message.field_by_name(orig_param)
            except KeyError:
                continue
            message_resource_map[orig_param] = getattr(ref, special_param, None)
        return message_resource_map

    def Parse(self, parent_params, parameter_info, aggregations_dict):
        """Parse the parent resource from parameter info and aggregations.

    Args:
      parent_params: [str], a list of params in the current collection's parent
        collection.
      parameter_info: the runtime ResourceParameterInfo object.
      aggregations_dict: {str: str}, a dict of params to values that are
        being aggregated from earlier updates.

    Returns:
      resources.Resource | None, the parsed parent reference or None if there
        is not enough information to parse.
    """
        param_values = {self.param_translation.get(p, p): parameter_info.GetValue(p) for p in parent_params}
        for p, value in six.iteritems(aggregations_dict):
            translated_name = self.param_translation.get(p, p)
            if value and (not param_values.get(translated_name, None)):
                param_values[translated_name] = value
        try:
            return resources.Resource(resources.REGISTRY, collection_info=resources.REGISTRY.GetCollectionInfo(self.collection), subcollection='', param_values=param_values, endpoint_url=None)
        except resources.Error as e:
            log.info(six.text_type(e).rstrip())
            return None