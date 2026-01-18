from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetLoggingMetadataArgDeprecated(messages):
    return arg_utils.ChoiceEnumMapper('--metadata', messages.SubnetworkLogConfig.MetadataValueValuesEnum, custom_mappings={'INCLUDE_ALL_METADATA': 'include-all-metadata', 'EXCLUDE_ALL_METADATA': 'exclude-all-metadata'}, help_str='        Can only be specified if VPC Flow Logs for this subnetwork is\n        enabled. Configures whether metadata fields should be added to the\n        reported logs. Default is to exclude all metadata.\n        ')