from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def GetNodePoolGetRequest(args, release_track):
    """Get node pool get request message.

  Args:
    args: comand line arguments.
    release_track: release track of the command.

  Returns:
    message obj, node pool get request message.
  """
    messages = util.GetMessagesModule(release_track)
    req = messages.EdgecontainerProjectsLocationsClustersNodePoolsGetRequest(name=args.CONCEPTS.node_pool.Parse().RelativeName())
    return req