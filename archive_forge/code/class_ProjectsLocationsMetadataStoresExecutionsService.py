from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMetadataStoresExecutionsService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores_executions resource."""
    _NAME = 'projects_locations_metadataStores_executions'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMetadataStoresExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def AddExecutionEvents(self, request, global_params=None):
        """Adds Events to the specified Execution. An Event indicates whether an Artifact was used as an input or output for an Execution. If an Event already exists between the Execution and the Artifact, the Event is skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsAddExecutionEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddExecutionEventsResponse) The response message.
      """
        config = self.GetMethodConfig('AddExecutionEvents')
        return self._RunMethod(config, request, global_params=global_params)
    AddExecutionEvents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions/{executionsId}:addExecutionEvents', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.executions.addExecutionEvents', ordered_params=['execution'], path_params=['execution'], query_params=[], relative_path='v1/{+execution}:addExecutionEvents', request_field='googleCloudAiplatformV1AddExecutionEventsRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsAddExecutionEventsRequest', response_type_name='GoogleCloudAiplatformV1AddExecutionEventsResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an Execution associated with a MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Execution) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.executions.create', ordered_params=['parent'], path_params=['parent'], query_params=['executionId'], relative_path='v1/{+parent}/executions', request_field='googleCloudAiplatformV1Execution', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsCreateRequest', response_type_name='GoogleCloudAiplatformV1Execution', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Execution.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions/{executionsId}', http_method='DELETE', method_id='aiplatform.projects.locations.metadataStores.executions.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific Execution.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Execution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions/{executionsId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.executions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsGetRequest', response_type_name='GoogleCloudAiplatformV1Execution', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Executions in the MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.executions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/executions', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsListRequest', response_type_name='GoogleCloudAiplatformV1ListExecutionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a stored Execution.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Execution) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions/{executionsId}', http_method='PATCH', method_id='aiplatform.projects.locations.metadataStores.executions.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Execution', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsPatchRequest', response_type_name='GoogleCloudAiplatformV1Execution', supports_download=False)

    def Purge(self, request, global_params=None):
        """Purges Executions.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsPurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Purge')
        return self._RunMethod(config, request, global_params=global_params)
    Purge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions:purge', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.executions.purge', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/executions:purge', request_field='googleCloudAiplatformV1PurgeExecutionsRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsPurgeRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def QueryExecutionInputsAndOutputs(self, request, global_params=None):
        """Obtains the set of input and output Artifacts for this Execution, in the form of LineageSubgraph that also contains the Execution and connecting Events.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsQueryExecutionInputsAndOutputsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
        config = self.GetMethodConfig('QueryExecutionInputsAndOutputs')
        return self._RunMethod(config, request, global_params=global_params)
    QueryExecutionInputsAndOutputs.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/executions/{executionsId}:queryExecutionInputsAndOutputs', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.executions.queryExecutionInputsAndOutputs', ordered_params=['execution'], path_params=['execution'], query_params=[], relative_path='v1/{+execution}:queryExecutionInputsAndOutputs', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresExecutionsQueryExecutionInputsAndOutputsRequest', response_type_name='GoogleCloudAiplatformV1LineageSubgraph', supports_download=False)