from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import subnets_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetPrivateIpv6GoogleAccessTypeFlagMapper(messages):
    return arg_utils.ChoiceEnumMapper('--private-ipv6-google-access-type', messages.Subnetwork.PrivateIpv6GoogleAccessValueValuesEnum, custom_mappings={'DISABLE_GOOGLE_ACCESS': 'disable', 'ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE': 'enable-bidirectional-access', 'ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE': 'enable-outbound-vm-access'}, help_str='The private IPv6 google access type for the VMs in this subnet.')