from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List
from googlecloudsdk.api_lib.vmware import clusters
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.command_lib.vmware.clusters import util
from googlecloudsdk.core import log
def _ParseOldNodesConfigsParameters(existing_cluster, nodes_configs):
    """Parses the node configs parameters passed in the old format.

  In the old format, the nodes configs are passed in a way that specifies what
  exact node configs should be attached to the cluster after the operation. It's
  not possible to remove existing node types. Even unchanged nodes configs have
  to be specified in the parameters.

  Args:
    existing_cluster: cluster whose nodes configs should be updated
    nodes_configs: nodes configs to be attached to the cluster

  Returns:
    list of NodeTypeConfig objects prepared for further processing

  Raises:
    InvalidNodeConfigsProvidedError:
      if duplicate node types were specified or if a config for an existing node
      type is not specified
  """
    current_node_types = [prop.key for prop in existing_cluster.nodeTypeConfigs.additionalProperties]
    requested_node_types = [config['type'] for config in nodes_configs]
    duplicated_types = util.FindDuplicatedTypes(requested_node_types)
    if duplicated_types:
        raise util.InvalidNodeConfigsProvidedError(f'types: {duplicated_types} provided more than once.')
    unspecified_types = set(current_node_types) - set(requested_node_types)
    if unspecified_types:
        raise util.InvalidNodeConfigsProvidedError(f'when using `--node-type-config` parameters you need to specify node counts for all node types present in the cluster. Missing node types: {list(unspecified_types)}.')
    return [util.NodeTypeConfig(type=config['type'], count=config['count'], custom_core_count=0) for config in nodes_configs]