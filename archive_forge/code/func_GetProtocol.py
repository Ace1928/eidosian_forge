from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetProtocol(self, protocol):
    protocol_enum = arg_utils.ChoiceEnumMapper(arg_name='protocol', message_enum=self.messages.LoggingServer.ProtocolValueValuesEnum, include_filter=lambda x: 'PROTOCOL_UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(protocol))
    return protocol_enum