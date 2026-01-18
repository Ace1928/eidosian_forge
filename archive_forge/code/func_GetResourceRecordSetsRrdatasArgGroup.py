from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetResourceRecordSetsRrdatasArgGroup(use_deprecated_names=False):
    """Returns arg group for rrdatas flags.

  Args:
    use_deprecated_names: If true, uses snake_case names for flags
      --routing-policy-type and --routing-policy-data, --routing_policy_type and
      --routing_policy_data.  This group is defined with required=True and
      mutex=True, meaning that exactly one of these two arg configurations must
      be specified: --rrdatas --routing-policy-type AND --routing-policy-data
  """
    primary_backup_data_group = base.ArgumentGroup(help='Configuration for primary backup routing policy')
    primary_backup_data_group.AddArgument(GetResourceRecordSetsRoutingPolicyPrimaryDataArg(required=True))
    primary_backup_data_group.AddArgument(GetResourceRecordSetsRoutingPolicyBackupDataArg(required=True))
    primary_backup_data_group.AddArgument(GetResourceRecordSetsRoutingPolicyBackupDataTypeArg(required=True))
    primary_backup_data_group.AddArgument(GetResourceRecordSetsBackupDataTrickleRatio(required=False))
    policy_data_group = base.ArgumentGroup(required=True, mutex=True, help='Routing policy data arguments. Combines routing-policy-data, routing-policy-primary-data, routing-policy-backup-data.')
    policy_data_group.AddArgument(GetResourceRecordSetsRoutingPolicyDataArg(deprecated_name=use_deprecated_names))
    policy_data_group.AddArgument(primary_backup_data_group)
    policy_group = base.ArgumentGroup(required=False, help='Routing policy arguments. If you specify one of --routing-policy-data or --routing-policy-type, you must specify both.')
    policy_group.AddArgument(GetResourceRecordSetsRoutingPolicyTypeArg(required=True, deprecated_name=use_deprecated_names))
    policy_group.AddArgument(GetResourceRecordSetsEnableFencingArg(required=False))
    policy_group.AddArgument(GetResourceRecordSetsEnableHealthChecking(required=False))
    policy_group.AddArgument(policy_data_group)
    rrdatas_group = base.ArgumentGroup(required=True, mutex=True, help='Resource record sets arguments. Can specify either --rrdatas or both --routing-policy-data and --routing-policy-type.')
    rrdatas_group.AddArgument(GetResourceRecordSetsRrdatasArg(required=False))
    rrdatas_group.AddArgument(policy_group)
    return rrdatas_group