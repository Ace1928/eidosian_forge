from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudshell.v1 import cloudshell_v1_messages as messages
class UsersService(base_api.BaseApiService):
    """Service class for the users resource."""
    _NAME = 'users'

    def __init__(self, client):
        super(CloudshellV1.UsersService, self).__init__(client)
        self._upload_configs = {}