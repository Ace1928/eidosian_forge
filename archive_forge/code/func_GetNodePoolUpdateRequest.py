from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def GetNodePoolUpdateRequest(args, release_track, existing_node_pool):
    """Get node pool update request message.

  Args:
    args: comand line arguments.
    release_track: release track of the command.
    existing_node_pool: existing node pool.

  Returns:
    message obj, node pool update request message.
  """
    messages = util.GetMessagesModule(release_track)
    req = messages.EdgecontainerProjectsLocationsClustersNodePoolsPatchRequest(name=args.CONCEPTS.node_pool.Parse().RelativeName(), nodePool=messages.NodePool())
    update_mask_pieces = []
    PopulateNodePoolUpdateMessage(req, messages, args, update_mask_pieces, existing_node_pool)
    req.updateMask = ','.join(update_mask_pieces)
    return req