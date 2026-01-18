from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsTopicsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations_topics resource."""
    _NAME = 'admin_projects_locations_topics'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsTopicsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Topic) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics', http_method='POST', method_id='pubsublite.admin.projects.locations.topics.create', ordered_params=['parent'], path_params=['parent'], query_params=['topicId'], relative_path='v1/admin/{+parent}/topics', request_field='topic', request_type_name='PubsubliteAdminProjectsLocationsTopicsCreateRequest', response_type_name='Topic', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}', http_method='DELETE', method_id='pubsublite.admin.projects.locations.topics.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsTopicsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the topic configuration.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Topic) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}', http_method='GET', method_id='pubsublite.admin.projects.locations.topics.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsTopicsGetRequest', response_type_name='Topic', supports_download=False)

    def GetPartitions(self, request, global_params=None):
        """Returns the partition information for the requested topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsGetPartitionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TopicPartitions) The response message.
      """
        config = self.GetMethodConfig('GetPartitions')
        return self._RunMethod(config, request, global_params=global_params)
    GetPartitions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}/partitions', http_method='GET', method_id='pubsublite.admin.projects.locations.topics.getPartitions', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}/partitions', request_field='', request_type_name='PubsubliteAdminProjectsLocationsTopicsGetPartitionsRequest', response_type_name='TopicPartitions', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of topics for the given project.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTopicsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics', http_method='GET', method_id='pubsublite.admin.projects.locations.topics.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/admin/{+parent}/topics', request_field='', request_type_name='PubsubliteAdminProjectsLocationsTopicsListRequest', response_type_name='ListTopicsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates properties of the specified topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Topic) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}', http_method='PATCH', method_id='pubsublite.admin.projects.locations.topics.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/admin/{+name}', request_field='topic', request_type_name='PubsubliteAdminProjectsLocationsTopicsPatchRequest', response_type_name='Topic', supports_download=False)