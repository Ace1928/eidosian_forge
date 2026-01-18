from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.essentialcontacts.v1 import essentialcontacts_v1_messages as messages
Allows a contact admin to send a test message to contact to verify that it has been configured correctly.

      Args:
        request: (EssentialcontactsProjectsContactsSendTestMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      