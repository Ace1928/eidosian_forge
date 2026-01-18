from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventarcpublishing.v1 import eventarcpublishing_v1_messages as messages
Publish events to a message bus.

      Args:
        request: (EventarcpublishingProjectsLocationsMessageBusesPublishRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEventarcPublishingV1PublishResponse) The response message.
      