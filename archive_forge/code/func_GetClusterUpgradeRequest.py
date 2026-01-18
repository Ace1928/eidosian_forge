from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import admin_users
from googlecloudsdk.command_lib.edge_cloud.container import fleet
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import resources
def GetClusterUpgradeRequest(args, release_track):
    """Get cluster upgrade request message.

  Args:
    args: comand line arguments.
    release_track: release track of the command.

  Returns:
    message obj, cluster upgrade request message.
  """
    messages = util.GetMessagesModule(release_track)
    cluster_ref = GetClusterReference(args)
    upgrade_cluster_req = messages.UpgradeClusterRequest()
    upgrade_cluster_req.targetVersion = args.version
    if args.schedule.upper() != 'IMMEDIATELY':
        raise ValueError('Unsupported --schedule value: ' + args.schedule)
    upgrade_cluster_req.schedule = messages.UpgradeClusterRequest.ScheduleValueValuesEnum(args.schedule.upper())
    req = messages.EdgecontainerProjectsLocationsClustersUpgradeRequest()
    req.name = cluster_ref.RelativeName()
    req.upgradeClusterRequest = upgrade_cluster_req
    return req