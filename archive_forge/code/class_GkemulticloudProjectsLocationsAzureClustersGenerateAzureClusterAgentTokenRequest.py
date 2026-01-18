from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersGenerateAzureClusterAgentTokenRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersGenerateAzureClusterAgentTo
  kenRequest object.

  Fields:
    azureCluster: Required.
    googleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest: A
      GoogleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest resource
      to be passed as the request body.
  """
    azureCluster = _messages.StringField(1, required=True)
    googleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest = _messages.MessageField('GoogleCloudGkemulticloudV1GenerateAzureClusterAgentTokenRequest', 2)