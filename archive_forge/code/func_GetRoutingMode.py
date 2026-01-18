from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.api_lib.vmware.networks import NetworksClient
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetRoutingMode(self, routing_mode):
    routing_mode_enum = arg_utils.ChoiceEnumMapper(arg_name='routing_mode', message_enum=self.messages.PrivateConnection.RoutingModeValueValuesEnum, include_filter=lambda x: 'ROUTING_MODE_UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(routing_mode))
    return routing_mode_enum