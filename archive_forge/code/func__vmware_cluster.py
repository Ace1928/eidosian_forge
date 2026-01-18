from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator, Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.api_lib.container.vmware import version_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _vmware_cluster(self, args: parser_extensions.Namespace):
    """Constructs proto message VmwareCluster."""
    kwargs = {'name': self._user_cluster_name(args), 'adminClusterMembership': self._admin_cluster_membership_name(args), 'description': flags.Get(args, 'description'), 'onPremVersion': flags.Get(args, 'version'), 'annotations': self._annotations(args), 'controlPlaneNode': self._vmware_control_plane_node_config(args), 'antiAffinityGroups': self._vmware_aag_config(args), 'storage': self._vmware_storage_config(args), 'networkConfig': self._vmware_network_config(args), 'loadBalancer': self._vmware_load_balancer_config(args), 'vcenter': self._vmware_vcenter_config(args), 'dataplaneV2': self._vmware_dataplane_v2_config(args), 'vmTrackingEnabled': self._vm_tracking_enabled(args), 'autoRepairConfig': self._vmware_auto_repair_config(args), 'authorization': self._authorization(args), 'enableControlPlaneV2': self._enable_control_plane_v2(args), 'upgradePolicy': self._upgrade_policy(args)}
    if any(kwargs.values()):
        return messages.VmwareCluster(**kwargs)
    return None