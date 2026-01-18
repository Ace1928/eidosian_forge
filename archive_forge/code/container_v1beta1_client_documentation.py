from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1beta1 import container_v1beta1_messages as messages
Returns configuration info about the Google Kubernetes Engine service.

      Args:
        request: (ContainerProjectsZonesGetServerconfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServerConfig) The response message.
      