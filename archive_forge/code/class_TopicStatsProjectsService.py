from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class TopicStatsProjectsService(base_api.BaseApiService):
    """Service class for the topicStats_projects resource."""
    _NAME = 'topicStats_projects'

    def __init__(self, client):
        super(PubsubliteV1.TopicStatsProjectsService, self).__init__(client)
        self._upload_configs = {}