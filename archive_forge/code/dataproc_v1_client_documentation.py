from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
Updates (replaces) workflow template. The updated template must contain version that matches the current server version.

      Args:
        request: (WorkflowTemplate) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkflowTemplate) The response message.
      