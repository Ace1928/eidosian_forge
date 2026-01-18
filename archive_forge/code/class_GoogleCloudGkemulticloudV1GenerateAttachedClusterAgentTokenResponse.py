from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenResponse(_messages.Message):
    """A GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenResponse
  object.

  Fields:
    access_token: A string attribute.
    expires_in: A integer attribute.
    token_type: A string attribute.
  """
    access_token = _messages.StringField(1)
    expires_in = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    token_type = _messages.StringField(3)