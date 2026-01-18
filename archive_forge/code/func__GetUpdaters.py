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
def _GetUpdaters(self):
    """Helper function to build dict of updaters."""
    final_param = self.collection_info.GetParams('')[-1]
    for i, attribute in enumerate(self.resource_spec.attributes):
        if self.resource_spec.ParamName(attribute.name) == final_param:
            attribute_idx = i
            break
    else:
        attribute_idx = 0
    updaters = {}
    for i, attribute in enumerate(self.resource_spec.attributes[:attribute_idx]):
        completer = CompleterForAttribute(self.resource_spec, attribute.name)
        if completer:
            updaters[self.resource_spec.ParamName(attribute.name)] = (completer, True)
        else:
            updaters[self.resource_spec.ParamName(attribute.name)] = (None, False)
    return updaters