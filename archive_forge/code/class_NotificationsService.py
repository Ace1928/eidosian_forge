import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class NotificationsService(base_api.BaseApiService):
    """Service class for the notifications resource."""
    _NAME = u'notifications'

    def __init__(self, client):
        super(StorageV1.NotificationsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Permanently deletes a notification subscription.

      Args:
        request: (StorageNotificationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageNotificationsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.notifications.delete', ordered_params=[u'bucket', u'notification'], path_params=[u'bucket', u'notification'], query_params=[u'userProject'], relative_path=u'b/{bucket}/notificationConfigs/{notification}', request_field='', request_type_name=u'StorageNotificationsDeleteRequest', response_type_name=u'StorageNotificationsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """View a notification configuration.

      Args:
        request: (StorageNotificationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Notification) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.notifications.get', ordered_params=[u'bucket', u'notification'], path_params=[u'bucket', u'notification'], query_params=[u'userProject'], relative_path=u'b/{bucket}/notificationConfigs/{notification}', request_field='', request_type_name=u'StorageNotificationsGetRequest', response_type_name=u'Notification', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a notification subscription for a given bucket.

      Args:
        request: (StorageNotificationsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Notification) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.notifications.insert', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'userProject'], relative_path=u'b/{bucket}/notificationConfigs', request_field=u'notification', request_type_name=u'StorageNotificationsInsertRequest', response_type_name=u'Notification', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of notification subscriptions for a given bucket.

      Args:
        request: (StorageNotificationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Notifications) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.notifications.list', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'userProject'], relative_path=u'b/{bucket}/notificationConfigs', request_field='', request_type_name=u'StorageNotificationsListRequest', response_type_name=u'Notifications', supports_download=False)