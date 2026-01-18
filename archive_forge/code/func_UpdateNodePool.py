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
def UpdateNodePool(self, node_pool_ref, options):
    if options.IsAutoscalingUpdate():
        autoscaling = self.UpdateNodePoolAutoscaling(node_pool_ref, options)
        update = self.messages.ClusterUpdate(desiredNodePoolId=node_pool_ref.nodePoolId, desiredNodePoolAutoscaling=autoscaling)
        operation = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId), update=update))
        return self.ParseOperation(operation.name, node_pool_ref.zone)
    elif options.IsNodePoolManagementUpdate():
        management = self.UpdateNodePoolNodeManagement(node_pool_ref, options)
        req = self.messages.SetNodePoolManagementRequest(name=ProjectLocationClusterNodePool(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId, node_pool_ref.nodePoolId), management=management)
        operation = self.client.projects_locations_clusters_nodePools.SetManagement(req)
    elif options.IsUpdateNodePoolRequest():
        req = self.UpdateNodePoolRequest(node_pool_ref, options)
        operation = self.client.projects_locations_clusters_nodePools.Update(req)
    else:
        raise util.Error('Unhandled node pool update mode')
    return self.ParseOperation(operation.name, node_pool_ref.zone)