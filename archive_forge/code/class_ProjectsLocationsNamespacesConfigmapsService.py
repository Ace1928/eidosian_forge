from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ProjectsLocationsNamespacesConfigmapsService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_configmaps resource."""
    _NAME = 'projects_locations_namespaces_configmaps'

    def __init__(self, client):
        super(AnthoseventsV1.ProjectsLocationsNamespacesConfigmapsService, self).__init__(client)
        self._upload_configs = {}

    def Patch(self, request, global_params=None):
        """Rpc to update a ConfigMap.

      Args:
        request: (AnthoseventsProjectsLocationsNamespacesConfigmapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/configmaps/{configmapsId}', http_method='PATCH', method_id='anthosevents.projects.locations.namespaces.configmaps.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='configMap', request_type_name='AnthoseventsProjectsLocationsNamespacesConfigmapsPatchRequest', response_type_name='ConfigMap', supports_download=False)

    def ReplaceConfigMap(self, request, global_params=None):
        """Rpc to replace a ConfigMap.

      Args:
        request: (AnthoseventsProjectsLocationsNamespacesConfigmapsReplaceConfigMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('ReplaceConfigMap')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceConfigMap.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/configmaps/{configmapsId}', http_method='PUT', method_id='anthosevents.projects.locations.namespaces.configmaps.replaceConfigMap', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='configMap', request_type_name='AnthoseventsProjectsLocationsNamespacesConfigmapsReplaceConfigMapRequest', response_type_name='ConfigMap', supports_download=False)