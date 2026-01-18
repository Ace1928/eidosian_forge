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
def _ParseNewNodesConfigsParameters(existing_cluster, updated_nodes_configs, removed_types):
    """Parses the node configs parameters passed in the new format.

  In the new format, the nodes configs are passed using two parameters. One of
  them specifies which configs should be updated or created (unchanged configs
  don't have to be specified at all). The other lists the configs to be removed.
  This format is more flexible than the old one because it allows for config
  removal and doesn't require re-specifying unchanged configs.

  Args:
    existing_cluster: cluster whose nodes configs should be updated
    updated_nodes_configs: list of nodes configs to update or create
    removed_types: list of node types for which nodes configs should be removed

  Returns:
    list of NodeTypeConfig objects prepared for further processing

  Raises:
    InvalidNodeConfigsProvidedError:
      if duplicate node types were specified
  """
    requested_node_types = [config['type'] for config in updated_nodes_configs] + removed_types
    duplicated_types = util.FindDuplicatedTypes(requested_node_types)
    if duplicated_types:
        raise util.InvalidNodeConfigsProvidedError(f'types: {duplicated_types} provided more than once.')
    node_count = {}
    for prop in existing_cluster.nodeTypeConfigs.additionalProperties:
        node_count[prop.key] = prop.value.nodeCount
    for config in updated_nodes_configs:
        node_count[config['type']] = config['count']
    for node_type in removed_types:
        node_count[node_type] = 0
    return [util.NodeTypeConfig(type=node_type, count=count, custom_core_count=0) for node_type, count in node_count.items()]