from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionDiskTypesService(base_api.BaseApiService):
    """Service class for the regionDiskTypes resource."""
    _NAME = 'regionDiskTypes'

    def __init__(self, client):
        super(ComputeBeta.RegionDiskTypesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the specified regional disk type.

      Args:
        request: (ComputeRegionDiskTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiskType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionDiskTypes.get', ordered_params=['project', 'region', 'diskType'], path_params=['diskType', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/diskTypes/{diskType}', request_field='', request_type_name='ComputeRegionDiskTypesGetRequest', response_type_name='DiskType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of regional disk types available to the specified project.

      Args:
        request: (ComputeRegionDiskTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionDiskTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionDiskTypes.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/diskTypes', request_field='', request_type_name='ComputeRegionDiskTypesListRequest', response_type_name='RegionDiskTypeList', supports_download=False)