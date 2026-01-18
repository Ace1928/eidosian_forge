from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policyanalyzer.v1 import policyanalyzer_v1_messages as messages
class ProjectsLocationsActivityTypesActivitiesService(base_api.BaseApiService):
    """Service class for the projects_locations_activityTypes_activities resource."""
    _NAME = 'projects_locations_activityTypes_activities'

    def __init__(self, client):
        super(PolicyanalyzerV1.ProjectsLocationsActivityTypesActivitiesService, self).__init__(client)
        self._upload_configs = {}

    def Query(self, request, global_params=None):
        """Queries policy activities on Google Cloud resources.

      Args:
        request: (PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicyanalyzerV1QueryActivityResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/activityTypes/{activityTypesId}/activities:query', http_method='GET', method_id='policyanalyzer.projects.locations.activityTypes.activities.query', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/activities:query', request_field='', request_type_name='PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest', response_type_name='GoogleCloudPolicyanalyzerV1QueryActivityResponse', supports_download=False)