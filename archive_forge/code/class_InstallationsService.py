from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class InstallationsService(base_api.BaseApiService):
    """Service class for the installations resource."""
    _NAME = 'installations'

    def __init__(self, client):
        super(CloudbuildV1.InstallationsService, self).__init__(client)
        self._upload_configs = {}