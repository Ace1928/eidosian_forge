from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def GetGKEClusterResoureLinkAndURI(gke_uri, gke_cluster):
    """Get GKE cluster's full resource name and cluster URI."""
    if gke_uri is None and gke_cluster is None:
        return (None, None)
    cluster_project = None
    if gke_uri:
        cluster_project, location, name = ParseGKEURI(gke_uri)
    else:
        cluster_project = properties.VALUES.core.project.GetOrFail()
        location, name = ParseGKECluster(gke_cluster)
    return ConstructGKEClusterResourceLinkAndURI(cluster_project, location, name)