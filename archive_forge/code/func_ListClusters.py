from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.container.fleet import util as hub_util
from googlecloudsdk.api_lib.resourcesettings import service as resourcesettings_service
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.runtime_config import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ListClusters(location=None, project=None):
    """Get all clusters with Cloud Run enabled.

  Args:
    location: str optional name of location to search for clusters in. Leaving
      this field blank will search in all locations.
    project: str optional name of project to search for clusters in. Leaving
      this field blank will use the project defined by the corresponding
      property.

  Returns:
    List of googlecloudsdk.generated_clients.apis.container.CONTAINER_API_VERSION
    import container_CONTAINER_API_VERSION_messages.Cluster objects
  """
    container_api = container_api_adapter.NewAPIAdapter(CONTAINER_API_VERSION)
    if not project:
        project = properties.VALUES.core.project.Get(required=True)
    response = container_api.ListClusters(project, location)
    if response.missingZones:
        log.warning('The following cluster locations did not respond: {}. List results may be incomplete.'.format(', '.join(response.missingZones)))

    def _SortKey(cluster):
        return (cluster.zone, cluster.name)
    clusters = sorted(response.clusters, key=_SortKey)
    crfa_cluster_names = ListCloudRunForAnthosClusters(project)
    return [c for c in clusters if c.name in crfa_cluster_names or (c.addonsConfig.cloudRunConfig and (not c.addonsConfig.cloudRunConfig.disabled))]