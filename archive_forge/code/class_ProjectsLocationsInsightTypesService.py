from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class ProjectsLocationsInsightTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_insightTypes resource."""
    _NAME = 'projects_locations_insightTypes'

    def __init__(self, client):
        super(RecommenderV1alpha2.ProjectsLocationsInsightTypesService, self).__init__(client)
        self._upload_configs = {}