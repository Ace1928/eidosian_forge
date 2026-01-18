from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
def GetGkeClusterIsWorkloadIdentityEnabled(project, location, cluster):
    """Determines if the GKE cluster is Workload Identity enabled."""
    gke_cluster = _GetGkeCluster(project, location, cluster)
    workload_identity_config = gke_cluster.workloadIdentityConfig
    if not workload_identity_config:
        log.debug('GKE cluster does not have a workloadIdentityConfig.')
        return False
    workload_pool = workload_identity_config.workloadPool
    if not workload_pool:
        log.debug("GKE cluster's workloadPool is the empty string.")
        return False
    return True