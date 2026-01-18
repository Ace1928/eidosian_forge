from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventarcpublishing.v1 import eventarcpublishing_v1_messages as messages
class ProjectsLocationsMessageBusesService(base_api.BaseApiService):
    """Service class for the projects_locations_messageBuses resource."""
    _NAME = 'projects_locations_messageBuses'

    def __init__(self, client):
        super(EventarcpublishingV1.ProjectsLocationsMessageBusesService, self).__init__(client)
        self._upload_configs = {}

    def Publish(self, request, global_params=None):
        """Publish events to a message bus.

      Args:
        request: (EventarcpublishingProjectsLocationsMessageBusesPublishRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEventarcPublishingV1PublishResponse) The response message.
      """
        config = self.GetMethodConfig('Publish')
        return self._RunMethod(config, request, global_params=global_params)
    Publish.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/messageBuses/{messageBusesId}:publish', http_method='POST', method_id='eventarcpublishing.projects.locations.messageBuses.publish', ordered_params=['messageBus'], path_params=['messageBus'], query_params=[], relative_path='v1/{+messageBus}:publish', request_field='googleCloudEventarcPublishingV1PublishRequest', request_type_name='EventarcpublishingProjectsLocationsMessageBusesPublishRequest', response_type_name='GoogleCloudEventarcPublishingV1PublishResponse', supports_download=False)