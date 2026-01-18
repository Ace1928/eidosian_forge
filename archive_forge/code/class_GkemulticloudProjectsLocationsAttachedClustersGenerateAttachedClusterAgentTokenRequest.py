from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAttachedClustersGenerateAttachedClusterAgentTokenRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAttachedClustersGenerateAttachedClusterA
  gentTokenRequest object.

  Fields:
    attachedCluster: Required.
    googleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest: A
      GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest
      resource to be passed as the request body.
  """
    attachedCluster = _messages.StringField(1, required=True)
    googleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest = _messages.MessageField('GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest', 2)