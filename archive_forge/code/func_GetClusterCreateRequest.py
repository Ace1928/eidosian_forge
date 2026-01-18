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
def GetClusterCreateRequest(args, release_track):
    """Get cluster create request message.

  Args:
    args: comand line arguments.
    release_track: release track of the command.

  Returns:
    message obj, cluster create request message.
  """
    messages = util.GetMessagesModule(release_track)
    cluster_ref = GetClusterReference(args)
    req = messages.EdgecontainerProjectsLocationsClustersCreateRequest(cluster=messages.Cluster(), clusterId=cluster_ref.clustersId, parent=cluster_ref.Parent().RelativeName())
    PopulateClusterMessage(req, messages, args)
    if release_track == base.ReleaseTrack.ALPHA:
        PopulateClusterAlphaMessage(req, messages, args)
    return req