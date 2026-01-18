from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class CursorProjectsService(base_api.BaseApiService):
    """Service class for the cursor_projects resource."""
    _NAME = 'cursor_projects'

    def __init__(self, client):
        super(PubsubliteV1.CursorProjectsService, self).__init__(client)
        self._upload_configs = {}