from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class ResourcesCalendarsService(base_api.BaseApiService):
    """Service class for the resources_calendars resource."""
    _NAME = u'resources_calendars'

    def __init__(self, client):
        super(AdminDirectoryV1.ResourcesCalendarsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a calendar resource.

      Args:
        request: (DirectoryResourcesCalendarsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryResourcesCalendarsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.resources.calendars.delete', ordered_params=[u'customer', u'calendarResourceId'], path_params=[u'calendarResourceId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/calendars/{calendarResourceId}', request_field='', request_type_name=u'DirectoryResourcesCalendarsDeleteRequest', response_type_name=u'DirectoryResourcesCalendarsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a calendar resource.

      Args:
        request: (DirectoryResourcesCalendarsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (CalendarResource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.calendars.get', ordered_params=[u'customer', u'calendarResourceId'], path_params=[u'calendarResourceId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/calendars/{calendarResourceId}', request_field='', request_type_name=u'DirectoryResourcesCalendarsGetRequest', response_type_name=u'CalendarResource', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a calendar resource.

      Args:
        request: (DirectoryResourcesCalendarsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (CalendarResource) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.resources.calendars.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/calendars', request_field=u'calendarResource', request_type_name=u'DirectoryResourcesCalendarsInsertRequest', response_type_name=u'CalendarResource', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of calendar resources for an account.

      Args:
        request: (DirectoryResourcesCalendarsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (CalendarResources) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resources.calendars.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'maxResults', u'orderBy', u'pageToken', u'query'], relative_path=u'customer/{customer}/resources/calendars', request_field='', request_type_name=u'DirectoryResourcesCalendarsListRequest', response_type_name=u'CalendarResources', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a calendar resource.

This method supports patch semantics, meaning you only need to include the
fields you wish to update. Fields that are not present in the request will be
preserved. This method supports patch semantics.

      Args:
        request: (DirectoryResourcesCalendarsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (CalendarResource) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.resources.calendars.patch', ordered_params=[u'customer', u'calendarResourceId'], path_params=[u'calendarResourceId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/calendars/{calendarResourceId}', request_field=u'calendarResource', request_type_name=u'DirectoryResourcesCalendarsPatchRequest', response_type_name=u'CalendarResource', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a calendar resource.

This method supports patch semantics, meaning you only need to include the
fields you wish to update. Fields that are not present in the request will be
preserved.

      Args:
        request: (DirectoryResourcesCalendarsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (CalendarResource) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.resources.calendars.update', ordered_params=[u'customer', u'calendarResourceId'], path_params=[u'calendarResourceId', u'customer'], query_params=[], relative_path=u'customer/{customer}/resources/calendars/{calendarResourceId}', request_field=u'calendarResource', request_type_name=u'DirectoryResourcesCalendarsUpdateRequest', response_type_name=u'CalendarResource', supports_download=False)