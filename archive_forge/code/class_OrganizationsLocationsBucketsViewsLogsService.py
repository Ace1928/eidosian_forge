from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class OrganizationsLocationsBucketsViewsLogsService(base_api.BaseApiService):
    """Service class for the organizations_locations_buckets_views_logs resource."""
    _NAME = 'organizations_locations_buckets_views_logs'

    def __init__(self, client):
        super(LoggingV2.OrganizationsLocationsBucketsViewsLogsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the logs in projects, organizations, folders, or billing accounts. Only logs that have entries are listed.

      Args:
        request: (LoggingOrganizationsLocationsBucketsViewsLogsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLogsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/buckets/{bucketsId}/views/{viewsId}/logs', http_method='GET', method_id='logging.organizations.locations.buckets.views.logs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'resourceNames'], relative_path='v2/{+parent}/logs', request_field='', request_type_name='LoggingOrganizationsLocationsBucketsViewsLogsListRequest', response_type_name='ListLogsResponse', supports_download=False)