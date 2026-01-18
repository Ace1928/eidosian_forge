from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
def ConstructGKEClusterResourceLinkAndURI(project_id, cluster_location, cluster_name):
    """Constructs GKE URI and resource name from args and container endpoint.

  Args:
    project_id: the project identifier to which the cluster to be registered
      belongs.
    cluster_location: zone or region of the cluster.
    cluster_name: name of the cluster to be registered.

  Returns:
    GKE resource link: full resource name as per go/resource-names
      (including preceding slashes).
    GKE cluster URI: URI string looks in the format of
    https://container.googleapis.com/v1/
      projects/{projectID}/locations/{location}/clusters/{clusterName}.
  """
    container_endpoint = core_apis.GetEffectiveApiEndpoint('container', 'v1')
    if container_endpoint.endswith('/'):
        container_endpoint = container_endpoint[:-1]
    gke_resource_link = '//{}/projects/{}/locations/{}/clusters/{}'.format(container_endpoint.replace('https://', '', 1).replace('http://', '', 1), project_id, cluster_location, cluster_name)
    gke_cluster_uri = '{}/v1/projects/{}/locations/{}/clusters/{}'.format(container_endpoint, project_id, cluster_location, cluster_name)
    return (gke_resource_link, gke_cluster_uri)