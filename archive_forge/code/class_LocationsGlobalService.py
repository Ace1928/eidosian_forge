from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
class LocationsGlobalService(base_api.BaseApiService):
    """Service class for the locations_global resource."""
    _NAME = 'locations_global'

    def __init__(self, client):
        super(MonitoringV1.LocationsGlobalService, self).__init__(client)
        self._upload_configs = {}