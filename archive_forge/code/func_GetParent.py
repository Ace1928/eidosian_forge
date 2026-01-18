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
def GetParent(self, parameter_info, aggregations=None, parent_translator=None):
    """Gets the parent reference of the parsed parameters.

    Args:
      parameter_info: the runtime ResourceParameterInfo object.
      aggregations: a list of _RuntimeParameter objects.
      parent_translator: a ParentTranslator for translating to a special
        parent collection, if needed.

    Returns:
      googlecloudsdk.core.resources.Resource | None, the parent resource or None
        if no parent was found.
    """
    aggregations_dict = self._GetAggregationsValuesDict(aggregations)
    param_values = self._GetRawParamValuesForParent(parameter_info, aggregations_dict=aggregations_dict)
    try:
        if not parent_translator:
            return self._ParseDefaultParent(param_values)
        return parent_translator.Parse(self._ParentParams(), parameter_info, aggregations_dict)
    except resources.ParentCollectionResolutionException as e:
        log.info(six.text_type(e).rstrip())
        return None
    except resources.Error as e:
        log.info(six.text_type(e).rstrip())
        return None