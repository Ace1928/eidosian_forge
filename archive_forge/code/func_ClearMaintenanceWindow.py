from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.run import flags
def ClearMaintenanceWindow(ref, args, request):
    """Clears cluster.maintenance_policy.window in the request if --clear-maintenance-window flag is specified.

  Args:
    ref: reference to the cluster object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if not flags.FlagIsExplicitlySet(args, 'clear_maintenance_window'):
        return request
    if not args.clear_maintenance_window:
        raise exceptions.BadArgumentException('--no-clear-maintenance-window', 'The flag is not supported')
    if request.cluster is None:
        release_track = args.calliope_command.ReleaseTrack()
        request.cluster = util.GetMessagesModule(release_track).Cluster()
    if request.cluster.maintenancePolicy:
        if request.cluster.maintenancePolicy.maintenanceExclusions:
            raise exceptions.BadArgumentException('--clear-maintenance-window', 'Cannot clear a maintenance window if there are maintenance exclusions.')
    request.cluster.maintenancePolicy = None
    _AddFieldToUpdateMask('maintenancePolicy', request)
    return request