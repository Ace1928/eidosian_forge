from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.run import flags
def AddMaintenanceExclusionWindow(ref, args, request):
    """Adds a maintenance exclusion window to the cluster if relevant flags are set.

  Args:
    ref: reference to the cluster object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if not flags.FlagIsExplicitlySet(args, 'add_maintenance_exclusion_name') and (not flags.FlagIsExplicitlySet(args, 'add_maintenance_exclusion_start')) and (not flags.FlagIsExplicitlySet(args, 'add_maintenance_exclusion_end')):
        return request
    _CheckAddMaintenanceExclusionFlags(args)
    release_track = args.calliope_command.ReleaseTrack()
    if request.cluster is None:
        request.cluster = util.GetMessagesModule(release_track).Cluster()
    if request.cluster.maintenancePolicy:
        for mew in request.cluster.maintenancePolicy.maintenanceExclusions:
            if args.add_maintenance_exclusion_name == mew.id:
                raise exceptions.BadArgumentException('--add-maintenance-exclusion-name', 'Maintenance exclusion name ' + mew.id + ' already exists.')
    request = RequestWithNewMaintenanceExclusion(request, util.GetMessagesModule(release_track), args)
    _AddFieldToUpdateMask('maintenancePolicy', request)
    return request