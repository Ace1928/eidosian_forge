from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsOptimizedHostStatsService(base_api.BaseApiService):
    """Service class for the organizations_optimizedHostStats resource."""
    _NAME = 'organizations_optimizedHostStats'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsOptimizedHostStatsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Similar to GetHostStats except that the response is less verbose.

      Args:
        request: (ApigeeOrganizationsOptimizedHostStatsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1OptimizedStats) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/optimizedHostStats/{optimizedHostStatsId}', http_method='GET', method_id='apigee.organizations.optimizedHostStats.get', ordered_params=['name'], path_params=['name'], query_params=['accuracy', 'envgroupHostname', 'filter', 'limit', 'offset', 'realtime', 'select', 'sort', 'sortby', 'timeRange', 'timeUnit', 'topk', 'tsAscending', 'tzo'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsOptimizedHostStatsGetRequest', response_type_name='GoogleCloudApigeeV1OptimizedStats', supports_download=False)