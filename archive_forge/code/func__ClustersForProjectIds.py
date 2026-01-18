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
def _ClustersForProjectIds(project_ids, cluster_location):
    response = []
    for project_id in project_ids:
        clusters = ListClusters(cluster_location, project_id)
        for cluster in clusters:
            response.append(GetClusterRef(cluster, project_id))
    return response