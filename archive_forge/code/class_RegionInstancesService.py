from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionInstancesService(base_api.BaseApiService):
    """Service class for the regionInstances resource."""
    _NAME = 'regionInstances'

    def __init__(self, client):
        super(ComputeBeta.RegionInstancesService, self).__init__(client)
        self._upload_configs = {}

    def BulkInsert(self, request, global_params=None):
        """Creates multiple instances in a given region. Count specifies the number of instances to create.

      Args:
        request: (ComputeRegionInstancesBulkInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BulkInsert')
        return self._RunMethod(config, request, global_params=global_params)
    BulkInsert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionInstances.bulkInsert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/instances/bulkInsert', request_field='bulkInsertInstanceResource', request_type_name='ComputeRegionInstancesBulkInsertRequest', response_type_name='Operation', supports_download=False)