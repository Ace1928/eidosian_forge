from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
Updates the Settings for the Project.

      Args:
        request: (ArtifactregistryProjectsUpdateProjectSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectSettings) The response message.
      