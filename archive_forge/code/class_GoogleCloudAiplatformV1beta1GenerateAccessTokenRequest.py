from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerateAccessTokenRequest(_messages.Message):
    """Request message for NotebookInternalService.GenerateAccessToken.

  Fields:
    vmToken: Required. The VM identity token (a JWT) for authenticating the
      VM. https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """
    vmToken = _messages.StringField(1)