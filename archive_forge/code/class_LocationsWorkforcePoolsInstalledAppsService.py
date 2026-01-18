from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsInstalledAppsService(base_api.BaseApiService):
    """Service class for the locations_workforcePools_installedApps resource."""
    _NAME = 'locations_workforcePools_installedApps'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsInstalledAppsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkforcePoolInstalledApp in a WorkforcePool. You cannot reuse the name of a deleted workforce pool installed app until 30 days after deletion.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolInstalledApp) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps', http_method='POST', method_id='iam.locations.workforcePools.installedApps.create', ordered_params=['parent'], path_params=['parent'], query_params=['workforcePoolInstalledAppId'], relative_path='v1/{+parent}/installedApps', request_field='workforcePoolInstalledApp', request_type_name='IamLocationsWorkforcePoolsInstalledAppsCreateRequest', response_type_name='WorkforcePoolInstalledApp', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a WorkforcePoolInstalledApp. You can undelete a workforce pool installed app for 30 days. After 30 days, deletion is permanent. You cannot update deleted workforce pool installed apps. However, you can view and list them.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolInstalledApp) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps/{installedAppsId}', http_method='DELETE', method_id='iam.locations.workforcePools.installedApps.delete', ordered_params=['name'], path_params=['name'], query_params=['validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsInstalledAppsDeleteRequest', response_type_name='WorkforcePoolInstalledApp', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an individual WorkforcePoolInstalledApp.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolInstalledApp) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps/{installedAppsId}', http_method='GET', method_id='iam.locations.workforcePools.installedApps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsInstalledAppsGetRequest', response_type_name='WorkforcePoolInstalledApp', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all non-deleted WorkforcePoolInstalledApps in a WorkforcePool][google.iam.admin.v1.WorkforcePool]. If `show_deleted` is set to `true`, then deleted installed apps are also listed.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkforcePoolInstalledAppsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps', http_method='GET', method_id='iam.locations.workforcePools.installedApps.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v1/{+parent}/installedApps', request_field='', request_type_name='IamLocationsWorkforcePoolsInstalledAppsListRequest', response_type_name='ListWorkforcePoolInstalledAppsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing WorkforcePoolInstalledApp.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolInstalledApp) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps/{installedAppsId}', http_method='PATCH', method_id='iam.locations.workforcePools.installedApps.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='workforcePoolInstalledApp', request_type_name='IamLocationsWorkforcePoolsInstalledAppsPatchRequest', response_type_name='WorkforcePoolInstalledApp', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkforcePoolInstalledApp, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamLocationsWorkforcePoolsInstalledAppsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkforcePoolInstalledApp) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/installedApps/{installedAppsId}:undelete', http_method='POST', method_id='iam.locations.workforcePools.installedApps.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkforcePoolInstalledAppRequest', request_type_name='IamLocationsWorkforcePoolsInstalledAppsUndeleteRequest', response_type_name='WorkforcePoolInstalledApp', supports_download=False)