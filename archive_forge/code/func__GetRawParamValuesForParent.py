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
def _GetRawParamValuesForParent(self, parameter_info, aggregations_dict=None):
    """Get raw param values for the resource in prep for parsing parent."""
    param_values = {p: parameter_info.GetValue(p) for p in self._ParentParams()}
    for name, value in six.iteritems(aggregations_dict or {}):
        if value and (not param_values.get(name, None)):
            param_values[name] = value
    final_param = self.collection_info.GetParams('')[-1]
    if param_values.get(final_param, None) is None:
        param_values[final_param] = 'fake'
    return param_values