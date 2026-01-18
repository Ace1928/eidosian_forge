from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.core import log
def _GetGkeCluster(project, location, cluster):
    """Gets the GKE cluster."""
    gke_client = gke_api_adapter.NewV1APIAdapter()
    try:
        return gke_client.GetCluster(gke_client.ParseCluster(name=cluster, location=location, project=project))
    except Exception as e:
        raise exceptions.GkeClusterGetError(e)