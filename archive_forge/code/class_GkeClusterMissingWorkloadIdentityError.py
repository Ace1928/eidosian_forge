from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class GkeClusterMissingWorkloadIdentityError(Error):
    """GKE Cluster is not Workload Identity enabled."""

    def __init__(self, gke_cluster_ref):
        super(GkeClusterMissingWorkloadIdentityError, self).__init__()
        self.gke_cluster_ref = gke_cluster_ref

    def __str__(self):
        return 'GKE Cluster "{0}" does not have Workload Identity enabled. Dataproc on GKE requires the GKE Cluster to have Workload Identity enabled. See https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity'.format(self.gke_cluster_ref.RelativeName())