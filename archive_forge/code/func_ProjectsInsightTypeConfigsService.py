from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def ProjectsInsightTypeConfigsService(api_version):
    """Returns the service class for the Project insight type configs."""
    client = RecommenderClient(api_version)
    return client.projects_locations_insightTypes