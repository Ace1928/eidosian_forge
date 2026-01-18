from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ProjectsLocationsNamespacesServiceaccountsService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_serviceaccounts resource."""
    _NAME = 'projects_locations_namespaces_serviceaccounts'

    def __init__(self, client):
        super(AnthoseventsV1.ProjectsLocationsNamespacesServiceaccountsService, self).__init__(client)
        self._upload_configs = {}

    def Patch(self, request, global_params=None):
        """Rpc to update Service Account.

      Args:
        request: (AnthoseventsProjectsLocationsNamespacesServiceaccountsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/serviceaccounts/{serviceaccountsId}', http_method='PATCH', method_id='anthosevents.projects.locations.namespaces.serviceaccounts.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='serviceAccount', request_type_name='AnthoseventsProjectsLocationsNamespacesServiceaccountsPatchRequest', response_type_name='ServiceAccount', supports_download=False)

    def ReplaceServiceAccount(self, request, global_params=None):
        """Rpc to replace a Service Account.

      Args:
        request: (AnthoseventsProjectsLocationsNamespacesServiceaccountsReplaceServiceAccountRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('ReplaceServiceAccount')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceServiceAccount.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/serviceaccounts/{serviceaccountsId}', http_method='PUT', method_id='anthosevents.projects.locations.namespaces.serviceaccounts.replaceServiceAccount', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='serviceAccount', request_type_name='AnthoseventsProjectsLocationsNamespacesServiceaccountsReplaceServiceAccountRequest', response_type_name='ServiceAccount', supports_download=False)