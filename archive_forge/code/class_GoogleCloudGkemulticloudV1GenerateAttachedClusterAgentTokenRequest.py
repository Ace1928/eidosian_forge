from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest(_messages.Message):
    """A GoogleCloudGkemulticloudV1GenerateAttachedClusterAgentTokenRequest
  object.

  Fields:
    audience: Optional.
    grantType: Optional.
    options: Optional.
    requestedTokenType: Optional.
    scope: Optional.
    subjectToken: Required.
    subjectTokenType: Required.
    version: Required.
  """
    audience = _messages.StringField(1)
    grantType = _messages.StringField(2)
    options = _messages.StringField(3)
    requestedTokenType = _messages.StringField(4)
    scope = _messages.StringField(5)
    subjectToken = _messages.StringField(6)
    subjectTokenType = _messages.StringField(7)
    version = _messages.StringField(8)