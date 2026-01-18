from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class OrganizationsTimeSeriesService(base_api.BaseApiService):
    """Service class for the organizations_timeSeries resource."""
    _NAME = 'organizations_timeSeries'

    def __init__(self, client):
        super(MonitoringV3.OrganizationsTimeSeriesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists time series that match a filter.

      Args:
        request: (MonitoringOrganizationsTimeSeriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTimeSeriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/organizations/{organizationsId}/timeSeries', http_method='GET', method_id='monitoring.organizations.timeSeries.list', ordered_params=['name'], path_params=['name'], query_params=['aggregation_alignmentPeriod', 'aggregation_crossSeriesReducer', 'aggregation_groupByFields', 'aggregation_perSeriesAligner', 'filter', 'interval_endTime', 'interval_startTime', 'orderBy', 'pageSize', 'pageToken', 'secondaryAggregation_alignmentPeriod', 'secondaryAggregation_crossSeriesReducer', 'secondaryAggregation_groupByFields', 'secondaryAggregation_perSeriesAligner', 'view'], relative_path='v3/{+name}/timeSeries', request_field='', request_type_name='MonitoringOrganizationsTimeSeriesListRequest', response_type_name='ListTimeSeriesResponse', supports_download=False)