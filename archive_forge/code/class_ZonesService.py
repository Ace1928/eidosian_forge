from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ZonesService(base_api.BaseApiService):
    """Service class for the zones resource."""
    _NAME = 'zones'

    def __init__(self, client):
        super(ComputeBeta.ZonesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the specified Zone resource.

      Args:
        request: (ComputeZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Zone) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zones.get', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}', request_field='', request_type_name='ComputeZonesGetRequest', response_type_name='Zone', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of Zone resources available to the specified project.

      Args:
        request: (ComputeZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ZoneList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zones.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones', request_field='', request_type_name='ComputeZonesListRequest', response_type_name='ZoneList', supports_download=False)