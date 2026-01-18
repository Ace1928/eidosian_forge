from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateNetworkConfigMessage(args, messages):
    """Creates the network config for the instance.

  Args:
    args: Argparse object from Command.Run
    messages: Module containing messages definition for the specified API.

  Returns:
    Network config for the instance.
  """
    network_config = messages.NetworkInterface
    network_name = None
    subnet_name = None
    nic_type = None
    if args.IsSpecified('network'):
        network_name = GetNetworkRelativeName(args)
    if args.IsSpecified('subnet'):
        subnet_name = GetSubnetRelativeName(args)
    if args.IsSpecified('nic_type'):
        nic_type = arg_utils.ChoiceEnumMapper(arg_name='nic-type', message_enum=network_config.NicTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.nic_type))
    return network_config(network=network_name, subnet=subnet_name, nicType=nic_type)