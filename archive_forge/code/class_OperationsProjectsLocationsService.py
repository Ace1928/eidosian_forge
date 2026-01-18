from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.videointelligence.v1 import videointelligence_v1_messages as messages
class OperationsProjectsLocationsService(base_api.BaseApiService):
    """Service class for the operations_projects_locations resource."""
    _NAME = 'operations_projects_locations'

    def __init__(self, client):
        super(VideointelligenceV1.OperationsProjectsLocationsService, self).__init__(client)
        self._upload_configs = {}