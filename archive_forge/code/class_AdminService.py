from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminService(base_api.BaseApiService):
    """Service class for the admin resource."""
    _NAME = 'admin'

    def __init__(self, client):
        super(PubsubliteV1.AdminService, self).__init__(client)
        self._upload_configs = {}