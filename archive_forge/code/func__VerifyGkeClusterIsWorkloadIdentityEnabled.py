from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import gke_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import gke_clusters
from googlecloudsdk.command_lib.dataproc import gke_workload_identity
from googlecloudsdk.command_lib.dataproc.gke_clusters import GkeNodePoolTargetsParser
from googlecloudsdk.core import log
@staticmethod
def _VerifyGkeClusterIsWorkloadIdentityEnabled(gke_cluster_ref):
    workload_identity_enabled = gke_helpers.GetGkeClusterIsWorkloadIdentityEnabled(project=gke_cluster_ref.projectsId, location=gke_cluster_ref.locationsId, cluster=gke_cluster_ref.clustersId)
    if not workload_identity_enabled:
        raise exceptions.GkeClusterMissingWorkloadIdentityError(gke_cluster_ref)