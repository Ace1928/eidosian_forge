from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class OrganizationsNotificationConfigsService(base_api.BaseApiService):
    """Service class for the organizations_notificationConfigs resource."""
    _NAME = 'organizations_notificationConfigs'

    def __init__(self, client):
        super(SecuritycenterV1.OrganizationsNotificationConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a notification config.

      Args:
        request: (SecuritycenterOrganizationsNotificationConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/notificationConfigs', http_method='POST', method_id='securitycenter.organizations.notificationConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['configId'], relative_path='v1/{+parent}/notificationConfigs', request_field='notificationConfig', request_type_name='SecuritycenterOrganizationsNotificationConfigsCreateRequest', response_type_name='NotificationConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a notification config.

      Args:
        request: (SecuritycenterOrganizationsNotificationConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/notificationConfigs/{notificationConfigsId}', http_method='DELETE', method_id='securitycenter.organizations.notificationConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsNotificationConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a notification config.

      Args:
        request: (SecuritycenterOrganizationsNotificationConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/notificationConfigs/{notificationConfigsId}', http_method='GET', method_id='securitycenter.organizations.notificationConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsNotificationConfigsGetRequest', response_type_name='NotificationConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists notification configs.

      Args:
        request: (SecuritycenterOrganizationsNotificationConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotificationConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/notificationConfigs', http_method='GET', method_id='securitycenter.organizations.notificationConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/notificationConfigs', request_field='', request_type_name='SecuritycenterOrganizationsNotificationConfigsListRequest', response_type_name='ListNotificationConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """ Updates a notification config. The following update fields are allowed: description, pubsub_topic, streaming_config.filter.

      Args:
        request: (SecuritycenterOrganizationsNotificationConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/notificationConfigs/{notificationConfigsId}', http_method='PATCH', method_id='securitycenter.organizations.notificationConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='notificationConfig', request_type_name='SecuritycenterOrganizationsNotificationConfigsPatchRequest', response_type_name='NotificationConfig', supports_download=False)