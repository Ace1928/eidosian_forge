from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.run import flags
def _CheckAddMaintenanceExclusionFlags(args):
    """Confirms all necessary flags for adding an exclusion window is set.

  Args:
    args: arguments passed through gcloud command

  Raises:
    BadArgumentException specifying missing flag
  """
    if not args.add_maintenance_exclusion_name:
        raise exceptions.BadArgumentException('--add-maintenance-exclusion-name', 'Every maintenance exclusion window must have a name.')
    if not args.add_maintenance_exclusion_start:
        raise exceptions.BadArgumentException('--add-maintenance-exclusion-start', 'Every maintenance exclusion window must have a start time.')
    if not args.add_maintenance_exclusion_end:
        raise exceptions.BadArgumentException('--add-maintenance-exclusion-end', 'Every maintenance exclusion window must have an end time.')