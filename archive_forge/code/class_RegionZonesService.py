from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionZonesService(base_api.BaseApiService):
    """Service class for the regionZones resource."""
    _NAME = 'regionZones'

    def __init__(self, client):
        super(ComputeBeta.RegionZonesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Retrieves the list of Zone resources under the specific region available to the specified project.

      Args:
        request: (ComputeRegionZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ZoneList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionZones.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/zones', request_field='', request_type_name='ComputeRegionZonesListRequest', response_type_name='ZoneList', supports_download=False)