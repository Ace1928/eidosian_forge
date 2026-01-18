from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ParseNodePools(self, options, node_config):
    """Creates a list of node pools for the cluster by parsing options.

    Args:
      options: cluster creation options
      node_config: node configuration for nodes in the node pools

    Returns:
      List of node pools.
    """
    max_nodes_per_pool = options.max_nodes_per_pool or MAX_NODES_PER_POOL
    num_pools = (options.num_nodes + max_nodes_per_pool - 1) // max_nodes_per_pool
    node_pool_name = options.node_pool_name or 'default-pool'
    if num_pools == 1:
        pool_names = [node_pool_name]
    else:
        pool_names = ['{0}-{1}'.format(node_pool_name, i) for i in range(0, num_pools)]
    pools = []
    nodes_per_pool = (options.num_nodes + num_pools - 1) // len(pool_names)
    to_add = options.num_nodes
    for name in pool_names:
        nodes = nodes_per_pool if to_add > nodes_per_pool else to_add
        pool = self.messages.NodePool(name=name, initialNodeCount=nodes, config=node_config, version=options.node_version, management=self._GetNodeManagement(options))
        if options.enable_autoscaling:
            pool.autoscaling = self.messages.NodePoolAutoscaling(enabled=options.enable_autoscaling, minNodeCount=options.min_nodes, maxNodeCount=options.max_nodes, totalMinNodeCount=options.total_min_nodes, totalMaxNodeCount=options.total_max_nodes)
            if options.location_policy is not None:
                pool.autoscaling.locationPolicy = LocationPolicyEnumFromString(self.messages, options.location_policy)
        if options.max_pods_per_node:
            if not options.enable_ip_alias:
                raise util.Error(MAX_PODS_PER_NODE_WITHOUT_IP_ALIAS_ERROR_MSG)
            pool.maxPodsConstraint = self.messages.MaxPodsConstraint(maxPodsPerNode=options.max_pods_per_node)
        if options.max_surge_upgrade is not None or options.max_unavailable_upgrade is not None:
            pool.upgradeSettings = self.messages.UpgradeSettings()
            pool.upgradeSettings.maxSurge = options.max_surge_upgrade
            pool.upgradeSettings.maxUnavailable = options.max_unavailable_upgrade
        if options.placement_type == 'COMPACT' or options.placement_policy is not None:
            pool.placementPolicy = self.messages.PlacementPolicy()
        if options.placement_type == 'COMPACT':
            pool.placementPolicy.type = self.messages.PlacementPolicy.TypeValueValuesEnum.COMPACT
        if options.placement_policy is not None:
            pool.placementPolicy.policyName = options.placement_policy
        if options.enable_queued_provisioning is not None:
            pool.queuedProvisioning = self.messages.QueuedProvisioning()
            pool.queuedProvisioning.enabled = options.enable_queued_provisioning
        pools.append(pool)
        to_add -= nodes
    return pools