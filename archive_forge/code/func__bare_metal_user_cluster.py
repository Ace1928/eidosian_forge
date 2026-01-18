from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bare_metal_user_cluster(self, args: parser_extensions.Namespace):
    """Constructs proto message Bare Metal Cluster."""
    kwargs = {'name': self._user_cluster_name(args), 'adminClusterMembership': self._admin_cluster_membership_name(args), 'description': getattr(args, 'description', None), 'annotations': self._annotations(args), 'bareMetalVersion': getattr(args, 'version', None), 'networkConfig': self._network_config(args), 'controlPlane': self._control_plane_config(args), 'loadBalancer': self._load_balancer_config(args), 'storage': self._storage_config(args), 'proxy': self._proxy_config(args), 'clusterOperations': self._cluster_operations_config(args), 'maintenanceConfig': self._maintenance_config(args), 'nodeConfig': self._workload_node_config(args), 'securityConfig': self._security_config(args), 'nodeAccessConfig': self._node_access_config(args), 'binaryAuthorization': self._binary_authorization(args), 'upgradePolicy': self._upgrade_policy(args)}
    if any(kwargs.values()):
        return messages.BareMetalCluster(**kwargs)
    return None