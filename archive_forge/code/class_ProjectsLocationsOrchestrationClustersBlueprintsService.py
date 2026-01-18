from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsOrchestrationClustersBlueprintsService(base_api.BaseApiService):
    """Service class for the projects_locations_orchestrationClusters_blueprints resource."""
    _NAME = 'projects_locations_orchestrationClusters_blueprints'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsOrchestrationClustersBlueprintsService, self).__init__(client)
        self._upload_configs = {}

    def Approve(self, request, global_params=None):
        """Approves a blueprint and commits a new revision.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsApproveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Approve')
        return self._RunMethod(config, request, global_params=global_params)
    Approve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}:approve', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.approve', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:approve', request_field='approveBlueprintRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsApproveRequest', response_type_name='Blueprint', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a blueprint.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.create', ordered_params=['parent'], path_params=['parent'], query_params=['blueprintId'], relative_path='v1/{+parent}/blueprints', request_field='blueprint', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsCreateRequest', response_type_name='Blueprint', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a blueprint and all its revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}', http_method='DELETE', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Discard(self, request, global_params=None):
        """Discards the changes in a blueprint and reverts the blueprint to the last approved blueprint revision. No changes take place if a blueprint does not have revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDiscardRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscardBlueprintChangesResponse) The response message.
      """
        config = self.GetMethodConfig('Discard')
        return self._RunMethod(config, request, global_params=global_params)
    Discard.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}:discard', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.discard', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:discard', request_field='discardBlueprintChangesRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDiscardRequest', response_type_name='DiscardBlueprintChangesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested blueprint.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsGetRequest', response_type_name='Blueprint', supports_download=False)

    def List(self, request, global_params=None):
        """List all blueprints.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBlueprintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/blueprints', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRequest', response_type_name='ListBlueprintsResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """List blueprint revisions of a given blueprint.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBlueprintRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}:listRevisions', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsListRevisionsRequest', response_type_name='ListBlueprintRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a blueprint.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}', http_method='PATCH', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='blueprint', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsPatchRequest', response_type_name='Blueprint', supports_download=False)

    def Propose(self, request, global_params=None):
        """Proposes a blueprint for approval of changes.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsProposeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Propose')
        return self._RunMethod(config, request, global_params=global_params)
    Propose.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}:propose', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.propose', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:propose', request_field='proposeBlueprintRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsProposeRequest', response_type_name='Blueprint', supports_download=False)

    def Reject(self, request, global_params=None):
        """Rejects a blueprint revision proposal and flips it back to Draft state.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsRejectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
        config = self.GetMethodConfig('Reject')
        return self._RunMethod(config, request, global_params=global_params)
    Reject.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints/{blueprintsId}:reject', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.reject', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:reject', request_field='rejectBlueprintRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsRejectRequest', response_type_name='Blueprint', supports_download=False)

    def SearchRevisions(self, request, global_params=None):
        """Searches across blueprint revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsSearchRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchBlueprintRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    SearchRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/blueprints:searchRevisions', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.blueprints.searchRevisions', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'query'], relative_path='v1/{+parent}/blueprints:searchRevisions', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsSearchRevisionsRequest', response_type_name='SearchBlueprintRevisionsResponse', supports_download=False)