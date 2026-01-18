from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
Lists a service project attachment for a given service project. You can call this API from any project to find if it is attached to a host project.

      Args:
        request: (ApphubProjectsLocationsLookupServiceProjectAttachmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupServiceProjectAttachmentResponse) The response message.
      