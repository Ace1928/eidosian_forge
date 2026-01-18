from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
class ProjectsLocationsService(base_api.BaseApiService):
    """Service class for the projects_locations resource."""
    _NAME = u'projects_locations'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get information about a location.

      Args:
        request: (CloudkmsProjectsLocationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Location) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}', http_method=u'GET', method_id=u'cloudkms.projects.locations.get', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='', request_type_name=u'CloudkmsProjectsLocationsGetRequest', response_type_name=u'Location', supports_download=False)

    def List(self, request, global_params=None):
        """Lists information about the supported locations for this service.

      Args:
        request: (CloudkmsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations', http_method=u'GET', method_id=u'cloudkms.projects.locations.list', ordered_params=[u'name'], path_params=[u'name'], query_params=[u'filter', u'pageSize', u'pageToken'], relative_path=u'v1/{+name}/locations', request_field='', request_type_name=u'CloudkmsProjectsLocationsListRequest', response_type_name=u'ListLocationsResponse', supports_download=False)