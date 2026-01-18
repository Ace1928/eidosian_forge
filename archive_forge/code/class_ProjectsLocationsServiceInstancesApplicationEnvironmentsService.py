from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataprocgdc.v1alpha1 import dataprocgdc_v1alpha1_messages as messages
class ProjectsLocationsServiceInstancesApplicationEnvironmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceInstances_applicationEnvironments resource."""
    _NAME = 'projects_locations_serviceInstances_applicationEnvironments'

    def __init__(self, client):
        super(DataprocgdcV1alpha1.ProjectsLocationsServiceInstancesApplicationEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an application environment associated with a Dataproc ServiceInstance.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApplicationEnvironment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/applicationEnvironments', http_method='POST', method_id='dataprocgdc.projects.locations.serviceInstances.applicationEnvironments.create', ordered_params=['parent'], path_params=['parent'], query_params=['applicationEnvironmentId', 'requestId'], relative_path='v1alpha1/{+parent}/applicationEnvironments', request_field='applicationEnvironment', request_type_name='DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsCreateRequest', response_type_name='ApplicationEnvironment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an application environment.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/applicationEnvironments/{applicationEnvironmentsId}', http_method='DELETE', method_id='dataprocgdc.projects.locations.serviceInstances.applicationEnvironments.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an application environment.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApplicationEnvironment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/applicationEnvironments/{applicationEnvironmentsId}', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.applicationEnvironments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsGetRequest', response_type_name='ApplicationEnvironment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists application environments in a location.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApplicationEnvironmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/applicationEnvironments', http_method='GET', method_id='dataprocgdc.projects.locations.serviceInstances.applicationEnvironments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/applicationEnvironments', request_field='', request_type_name='DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsListRequest', response_type_name='ListApplicationEnvironmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an application environment. Only supports updating state or labels.

      Args:
        request: (DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApplicationEnvironment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/serviceInstances/{serviceInstancesId}/applicationEnvironments/{applicationEnvironmentsId}', http_method='PATCH', method_id='dataprocgdc.projects.locations.serviceInstances.applicationEnvironments.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='applicationEnvironment', request_type_name='DataprocgdcProjectsLocationsServiceInstancesApplicationEnvironmentsPatchRequest', response_type_name='ApplicationEnvironment', supports_download=False)