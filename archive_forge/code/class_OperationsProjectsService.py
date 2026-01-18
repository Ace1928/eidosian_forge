from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.videointelligence.v1 import videointelligence_v1_messages as messages
class OperationsProjectsService(base_api.BaseApiService):
    """Service class for the operations_projects resource."""
    _NAME = 'operations_projects'

    def __init__(self, client):
        super(VideointelligenceV1.OperationsProjectsService, self).__init__(client)
        self._upload_configs = {}