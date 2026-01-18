from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def PopulateNodePoolCreateMessage(req, messages, args):
    """Fill the node pool message from command arguments.

  Args:
    req: create node pool request message.
    messages: message module of edgecontainer node pool.
    args: command line arguments.
  """
    req.nodePool.nodeCount = int(args.node_count)
    req.nodePool.nodeLocation = args.node_location
    if flags.FlagIsExplicitlySet(args, 'machine_filter'):
        req.nodePool.machineFilter = args.machine_filter
    if flags.FlagIsExplicitlySet(args, 'local_disk_kms_key'):
        req.nodePool.localDiskEncryption = messages.LocalDiskEncryption()
        req.nodePool.localDiskEncryption.kmsKey = args.local_disk_kms_key
    if flags.FlagIsExplicitlySet(args, 'labels'):
        req.nodePool.labels = messages.NodePool.LabelsValue()
        req.nodePool.labels.additionalProperties = []
        for key, value in args.labels.items():
            v = messages.NodePool.LabelsValue.AdditionalProperty()
            v.key = key
            v.value = value
            req.nodePool.labels.additionalProperties.append(v)
    if flags.FlagIsExplicitlySet(args, 'node_labels'):
        req.nodePool.nodeConfig = messages.NodeConfig()
        req.nodePool.nodeConfig.labels = messages.NodeConfig.LabelsValue()
        req.nodePool.nodeConfig.labels.additionalProperties = []
        for key, value in args.node_labels.items():
            v = messages.NodeConfig.LabelsValue.AdditionalProperty()
            v.key = key
            v.value = value
            req.nodePool.nodeConfig.labels.additionalProperties.append(v)