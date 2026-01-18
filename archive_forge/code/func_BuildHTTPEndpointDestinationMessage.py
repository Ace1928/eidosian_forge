from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def BuildHTTPEndpointDestinationMessage(self, destination_http_endpoint_uri, network_attachment):
    """Builds a HTTP Endpoint Destination message with the given data.

    Args:
      destination_http_endpoint_uri: str or None, the Trigger's destination uri.
      network_attachment: str or None, the Trigger's destination
        network attachment.

    Returns:
      A Destination message with a HTTP Endpoint destination.
    """
    http_endpoint_message = self._messages.HttpEndpoint(uri=destination_http_endpoint_uri)
    network_config_message = self._messages.NetworkConfig(networkAttachment=network_attachment)
    return self._messages.Destination(httpEndpoint=http_endpoint_message, networkConfig=network_config_message)