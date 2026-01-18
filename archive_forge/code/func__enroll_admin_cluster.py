from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.api_lib.container.gkeonprem import vmware_admin_clusters
from googlecloudsdk.api_lib.container.gkeonprem import vmware_clusters
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.gkeonprem import flags as common_flags
from googlecloudsdk.command_lib.container.vmware import constants
from googlecloudsdk.command_lib.container.vmware import errors
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import semver
def _enroll_admin_cluster(self, args, cluster_ref, admin_cluster_membership):
    admin_cluster_membership_ref = common_flags.GetAdminClusterMembershipResource(admin_cluster_membership)
    log.status.Print('Admin cluster is not enrolled. Enrolling admin cluster with membership [{}]'.format(admin_cluster_membership))
    admin_cluster_client = vmware_admin_clusters.AdminClustersClient()
    operation_client = operations.OperationsClient()
    operation = admin_cluster_client.Enroll(args, parent=cluster_ref.Parent().RelativeName(), membership=admin_cluster_membership, vmware_admin_cluster_id=admin_cluster_membership_ref.Name())
    operation_response = operation_client.Wait(operation)
    return operation_response