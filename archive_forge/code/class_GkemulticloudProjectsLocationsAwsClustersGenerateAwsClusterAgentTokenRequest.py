from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersGenerateAwsClusterAgentTokenRequest(_messages.Message):
    """A
  GkemulticloudProjectsLocationsAwsClustersGenerateAwsClusterAgentTokenRequest
  object.

  Fields:
    awsCluster: Required.
    googleCloudGkemulticloudV1GenerateAwsClusterAgentTokenRequest: A
      GoogleCloudGkemulticloudV1GenerateAwsClusterAgentTokenRequest resource
      to be passed as the request body.
  """
    awsCluster = _messages.StringField(1, required=True)
    googleCloudGkemulticloudV1GenerateAwsClusterAgentTokenRequest = _messages.MessageField('GoogleCloudGkemulticloudV1GenerateAwsClusterAgentTokenRequest', 2)