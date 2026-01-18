from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def GetNodePoolReference(args):
    """Get edgecontainer node pool resources.

  Args:
    args: command line arguments.

  Returns:
    edgecontainer node pool resources.
  """
    return resources.REGISTRY.ParseRelativeName(args.CONCEPTS.node_pool.Parse().RelativeName(), collection='edgecontainer.projects.locations.clusters.nodePools')