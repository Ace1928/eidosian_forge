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
def GetResourceRecordSetsRoutingPolicyBackupDataTypeArg(required=True):
    """Returns --routing_policy_backup_data_type command line arg value."""
    return base.Argument('--routing-policy-backup-data-type', metavar='ROUTING_POLICY_BACKUP_DATA_TYPE', required=required, choices=['GEO'], help='For FAILOVER routing policies, the type of routing policy the backup data uses. Currently, this must be GEO')