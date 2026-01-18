from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ApiV1Service(base_api.BaseApiService):
    """Service class for the api_v1 resource."""
    _NAME = 'api_v1'

    def __init__(self, client):
        super(RunV1.ApiV1Service, self).__init__(client)
        self._upload_configs = {}