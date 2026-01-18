from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.alpha import deploymentmanager_alpha_messages as messages
Lists all resource types for Deployment Manager.

      Args:
        request: (DeploymentmanagerTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypesListResponse) The response message.
      