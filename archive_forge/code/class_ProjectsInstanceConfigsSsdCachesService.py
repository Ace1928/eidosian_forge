from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstanceConfigsSsdCachesService(base_api.BaseApiService):
    """Service class for the projects_instanceConfigs_ssdCaches resource."""
    _NAME = 'projects_instanceConfigs_ssdCaches'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstanceConfigsSsdCachesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets information about a particular SSD cache.

      Args:
        request: (SpannerProjectsInstanceConfigsSsdCachesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SsdCache) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}/ssdCaches/{ssdCachesId}', http_method='GET', method_id='spanner.projects.instanceConfigs.ssdCaches.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstanceConfigsSsdCachesGetRequest', response_type_name='SsdCache', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all SSD caches for a given instance configurations.

      Args:
        request: (SpannerProjectsInstanceConfigsSsdCachesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSsdCachesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}/ssdCaches', http_method='GET', method_id='spanner.projects.instanceConfigs.ssdCaches.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/ssdCaches', request_field='', request_type_name='SpannerProjectsInstanceConfigsSsdCachesListRequest', response_type_name='ListSsdCachesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates SSD cache. The returned long-running operation can be used to track the progress of updating the SSD cache. If the named SSD cache does not exist, returns `NOT_FOUND`. While the operation is pending: * Cancelling the operation sets its metadata's cancel_time. It terminates with a `CANCELLED` status. * All other attempts to modify the SSD cache are rejected. * Reading the SSD cache via the API continues to give the pre-request values. Upon completion of the returned operation: * The SSD cache's new values are readable via the API. The returned long-running operation will have a name of the format `/operations/` and can be used to track the SSD cache modification. The metadata field type is UpdateSsdCacheMetadata. The response field type is SsdCache, if successful. Authorization requires `spanner.ssdCaches.update` permission on the resource name.

      Args:
        request: (SpannerProjectsInstanceConfigsSsdCachesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}/ssdCaches/{ssdCachesId}', http_method='PATCH', method_id='spanner.projects.instanceConfigs.ssdCaches.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='updateSsdCacheRequest', request_type_name='SpannerProjectsInstanceConfigsSsdCachesPatchRequest', response_type_name='Operation', supports_download=False)