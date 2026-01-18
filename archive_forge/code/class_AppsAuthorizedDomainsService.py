from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsAuthorizedDomainsService(base_api.BaseApiService):
    """Service class for the apps_authorizedDomains resource."""
    _NAME = 'apps_authorizedDomains'

    def __init__(self, client):
        super(AppengineV1beta.AppsAuthorizedDomainsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all domains the user is authorized to administer.

      Args:
        request: (AppengineAppsAuthorizedDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/authorizedDomains', http_method='GET', method_id='appengine.apps.authorizedDomains.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/authorizedDomains', request_field='', request_type_name='AppengineAppsAuthorizedDomainsListRequest', response_type_name='ListAuthorizedDomainsResponse', supports_download=False)