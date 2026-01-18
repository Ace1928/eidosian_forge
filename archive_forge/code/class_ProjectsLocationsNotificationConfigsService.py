from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class ProjectsLocationsNotificationConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_notificationConfigs resource."""
    _NAME = 'projects_locations_notificationConfigs'

    def __init__(self, client):
        super(SecuritycenterV2.ProjectsLocationsNotificationConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a notification config.

      Args:
        request: (SecuritycenterProjectsLocationsNotificationConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/notificationConfigs', http_method='POST', method_id='securitycenter.projects.locations.notificationConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['configId'], relative_path='v2/{+parent}/notificationConfigs', request_field='notificationConfig', request_type_name='SecuritycenterProjectsLocationsNotificationConfigsCreateRequest', response_type_name='NotificationConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a notification config.

      Args:
        request: (SecuritycenterProjectsLocationsNotificationConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/notificationConfigs/{notificationConfigsId}', http_method='DELETE', method_id='securitycenter.projects.locations.notificationConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterProjectsLocationsNotificationConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a notification config.

      Args:
        request: (SecuritycenterProjectsLocationsNotificationConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/notificationConfigs/{notificationConfigsId}', http_method='GET', method_id='securitycenter.projects.locations.notificationConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterProjectsLocationsNotificationConfigsGetRequest', response_type_name='NotificationConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists notification configs.

      Args:
        request: (SecuritycenterProjectsLocationsNotificationConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotificationConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/notificationConfigs', http_method='GET', method_id='securitycenter.projects.locations.notificationConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/notificationConfigs', request_field='', request_type_name='SecuritycenterProjectsLocationsNotificationConfigsListRequest', response_type_name='ListNotificationConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a notification config. The following update fields are allowed: description, pubsub_topic, streaming_config.filter.

      Args:
        request: (SecuritycenterProjectsLocationsNotificationConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/notificationConfigs/{notificationConfigsId}', http_method='PATCH', method_id='securitycenter.projects.locations.notificationConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='notificationConfig', request_type_name='SecuritycenterProjectsLocationsNotificationConfigsPatchRequest', response_type_name='NotificationConfig', supports_download=False)