from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsSecurityStatsService(base_api.BaseApiService):
    """Service class for the organizations_environments_securityStats resource."""
    _NAME = 'organizations_environments_securityStats'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsSecurityStatsService, self).__init__(client)
        self._upload_configs = {}

    def QueryTabularStats(self, request, global_params=None):
        """Retrieve security statistics as tabular rows.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityStatsQueryTabularStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1QueryTabularStatsResponse) The response message.
      """
        config = self.GetMethodConfig('QueryTabularStats')
        return self._RunMethod(config, request, global_params=global_params)
    QueryTabularStats.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityStats:queryTabularStats', http_method='POST', method_id='apigee.organizations.environments.securityStats.queryTabularStats', ordered_params=['orgenv'], path_params=['orgenv'], query_params=[], relative_path='v1/{+orgenv}/securityStats:queryTabularStats', request_field='googleCloudApigeeV1QueryTabularStatsRequest', request_type_name='ApigeeOrganizationsEnvironmentsSecurityStatsQueryTabularStatsRequest', response_type_name='GoogleCloudApigeeV1QueryTabularStatsResponse', supports_download=False)

    def QueryTimeSeriesStats(self, request, global_params=None):
        """Retrieve security statistics as a collection of time series.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityStatsQueryTimeSeriesStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1QueryTimeSeriesStatsResponse) The response message.
      """
        config = self.GetMethodConfig('QueryTimeSeriesStats')
        return self._RunMethod(config, request, global_params=global_params)
    QueryTimeSeriesStats.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityStats:queryTimeSeriesStats', http_method='POST', method_id='apigee.organizations.environments.securityStats.queryTimeSeriesStats', ordered_params=['orgenv'], path_params=['orgenv'], query_params=[], relative_path='v1/{+orgenv}/securityStats:queryTimeSeriesStats', request_field='googleCloudApigeeV1QueryTimeSeriesStatsRequest', request_type_name='ApigeeOrganizationsEnvironmentsSecurityStatsQueryTimeSeriesStatsRequest', response_type_name='GoogleCloudApigeeV1QueryTimeSeriesStatsResponse', supports_download=False)