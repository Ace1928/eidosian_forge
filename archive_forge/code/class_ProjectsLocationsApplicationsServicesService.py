from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
class ProjectsLocationsApplicationsServicesService(base_api.BaseApiService):
    """Service class for the projects_locations_applications_services resource."""
    _NAME = 'projects_locations_applications_services'

    def __init__(self, client):
        super(ApphubV1alpha.ProjectsLocationsApplicationsServicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Service in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsServicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/services', http_method='POST', method_id='apphub.projects.locations.applications.services.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceId'], relative_path='v1alpha/{+parent}/services', request_field='service', request_type_name='ApphubProjectsLocationsApplicationsServicesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Service from an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsServicesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/services/{servicesId}', http_method='DELETE', method_id='apphub.projects.locations.applications.services.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsApplicationsServicesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Service in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Service) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/services/{servicesId}', http_method='GET', method_id='apphub.projects.locations.applications.services.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsApplicationsServicesGetRequest', response_type_name='Service', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Services in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/services', http_method='GET', method_id='apphub.projects.locations.applications.services.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/services', request_field='', request_type_name='ApphubProjectsLocationsApplicationsServicesListRequest', response_type_name='ListServicesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Service in an Application.

      Args:
        request: (ApphubProjectsLocationsApplicationsServicesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/services/{servicesId}', http_method='PATCH', method_id='apphub.projects.locations.applications.services.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='service', request_type_name='ApphubProjectsLocationsApplicationsServicesPatchRequest', response_type_name='Operation', supports_download=False)