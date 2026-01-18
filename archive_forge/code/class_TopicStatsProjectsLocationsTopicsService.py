from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class TopicStatsProjectsLocationsTopicsService(base_api.BaseApiService):
    """Service class for the topicStats_projects_locations_topics resource."""
    _NAME = 'topicStats_projects_locations_topics'

    def __init__(self, client):
        super(PubsubliteV1.TopicStatsProjectsLocationsTopicsService, self).__init__(client)
        self._upload_configs = {}

    def ComputeHeadCursor(self, request, global_params=None):
        """Compute the head cursor for the partition. The head cursor's offset is guaranteed to be less than or equal to all messages which have not yet been acknowledged as published, and greater than the offset of any message whose publish has already been acknowledged. It is zero if there have never been messages in the partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeHeadCursorRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeHeadCursorResponse) The response message.
      """
        config = self.GetMethodConfig('ComputeHeadCursor')
        return self._RunMethod(config, request, global_params=global_params)
    ComputeHeadCursor.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/topicStats/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}:computeHeadCursor', http_method='POST', method_id='pubsublite.topicStats.projects.locations.topics.computeHeadCursor', ordered_params=['topic'], path_params=['topic'], query_params=[], relative_path='v1/topicStats/{+topic}:computeHeadCursor', request_field='computeHeadCursorRequest', request_type_name='PubsubliteTopicStatsProjectsLocationsTopicsComputeHeadCursorRequest', response_type_name='ComputeHeadCursorResponse', supports_download=False)

    def ComputeMessageStats(self, request, global_params=None):
        """Compute statistics about a range of messages in a given topic and partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeMessageStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeMessageStatsResponse) The response message.
      """
        config = self.GetMethodConfig('ComputeMessageStats')
        return self._RunMethod(config, request, global_params=global_params)
    ComputeMessageStats.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/topicStats/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}:computeMessageStats', http_method='POST', method_id='pubsublite.topicStats.projects.locations.topics.computeMessageStats', ordered_params=['topic'], path_params=['topic'], query_params=[], relative_path='v1/topicStats/{+topic}:computeMessageStats', request_field='computeMessageStatsRequest', request_type_name='PubsubliteTopicStatsProjectsLocationsTopicsComputeMessageStatsRequest', response_type_name='ComputeMessageStatsResponse', supports_download=False)

    def ComputeTimeCursor(self, request, global_params=None):
        """Compute the corresponding cursor for a publish or event time in a topic partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeTimeCursorRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeTimeCursorResponse) The response message.
      """
        config = self.GetMethodConfig('ComputeTimeCursor')
        return self._RunMethod(config, request, global_params=global_params)
    ComputeTimeCursor.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/topicStats/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}:computeTimeCursor', http_method='POST', method_id='pubsublite.topicStats.projects.locations.topics.computeTimeCursor', ordered_params=['topic'], path_params=['topic'], query_params=[], relative_path='v1/topicStats/{+topic}:computeTimeCursor', request_field='computeTimeCursorRequest', request_type_name='PubsubliteTopicStatsProjectsLocationsTopicsComputeTimeCursorRequest', response_type_name='ComputeTimeCursorResponse', supports_download=False)