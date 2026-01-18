from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def GetAvailabilityDomainScopeFlagMapper(messages):
    """Gets availability domain scope flag mapper for resource policies."""
    custom_mappings = {'UNSPECIFIED_SCOPE': ('unspecified-scope', 'Instances will be spread across different instrastructure to not share power, host and networking.'), 'HOST': ('host', 'Specifies availability domain scope across hosts. Instances will be spread across different hosts.')}
    return arg_utils.ChoiceEnumMapper('--scope', messages.ResourcePolicyGroupPlacementPolicy.ScopeValueValuesEnum, custom_mappings=custom_mappings, default=None, help_str='Scope specifies the availability domain to which the VMs should be spread.')