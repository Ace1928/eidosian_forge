from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudtasks.v2 import cloudtasks_v2_messages as messages
Creates or Updates a CMEK config. Updates the Customer Managed Encryption Key assotiated with the Cloud Tasks location (Creates if the key does not already exist). All new tasks created in the location will be encrypted at-rest with the KMS-key provided in the config.

      Args:
        request: (CloudtasksProjectsLocationsUpdateCmekConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CmekConfig) The response message.
      