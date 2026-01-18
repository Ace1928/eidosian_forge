from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddRestoreClusterSourceFlags(parser):
    """Adds RestoreCluster flags.

  Args:
    parser: argparse.ArgumentParser: Parser object for command line inputs.
  """
    group = parser.add_group(mutex=True, required=True, help='RestoreCluster source types.')
    group.add_argument('--backup', type=str, help='AlloyDB backup to restore from. This must either be the full backup name (projects/myProject/locations/us-central1/backups/myBackup) or the backup ID (myBackup). In the second case, the project and location are assumed to be the same as the restored cluster that is being created.')
    continuous_backup_source_group = group.add_group(help='Restore a cluster from a source cluster at a given point in time.')
    continuous_backup_source_group.add_argument('--source-cluster', required=True, help='AlloyDB source cluster to restore from. This must either be the full cluster name (projects/myProject/locations/us-central1/backups/myCluster) or the cluster ID (myCluster). In the second case, the project and location are assumed to be the same as the restored cluster that is being created.')
    continuous_backup_source_group.add_argument('--point-in-time', type=arg_parsers.Datetime.Parse, required=True, help='Point in time to restore to, in RFC 3339 format. For example, 2012-11-15T16:19:00.094Z.')