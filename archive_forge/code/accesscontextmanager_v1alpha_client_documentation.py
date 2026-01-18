from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
Lists all VPC-SC supported services.

      Args:
        request: (AccesscontextmanagerServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSupportedServicesResponse) The response message.
      