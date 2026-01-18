from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
def PopulateNodePoolUpdateMessage(req, messages, args, update_mask_pieces, existing_node_pool):
    """Fill the node pool message from command arguments.

  Args:
    req: update node pool request message.
    messages: message module of edgecontainer node pool.
    args: command line arguments.
    update_mask_pieces: update mask pieces.
    existing_node_pool: existing node pool.
  """
    if flags.FlagIsExplicitlySet(args, 'machine_filter'):
        update_mask_pieces.append('machineFilter')
        req.nodePool.machineFilter = args.machine_filter
    if flags.FlagIsExplicitlySet(args, 'node_count'):
        update_mask_pieces.append('nodeCount')
        req.nodePool.nodeCount = int(args.node_count)
    add_labels = labels_util.GetUpdateLabelsDictFromArgs(args)
    remove_labels = labels_util.GetRemoveLabelsListFromArgs(args)
    value_type = messages.NodePool.LabelsValue
    label_update_result = labels_util.Diff(additions=add_labels, subtractions=remove_labels, clear=args.clear_labels).Apply(value_type, existing_node_pool.labels)
    if label_update_result.needs_update:
        update_mask_pieces.append('labels')
        req.nodePool.labels = label_update_result.labels
    if flags.FlagIsExplicitlySet(args, 'node_labels'):
        update_mask_pieces.append('nodeConfig.labels')
        req.nodePool.nodeConfig = messages.NodeConfig()
        req.nodePool.nodeConfig.labels = messages.NodeConfig.LabelsValue()
        req.nodePool.nodeConfig.labels.additionalProperties = []
        for key, value in args.node_labels.items():
            v = messages.NodeConfig.LabelsValue.AdditionalProperty()
            v.key = key
            v.value = value
            req.nodePool.nodeConfig.labels.additionalProperties.append(v)