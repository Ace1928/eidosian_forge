from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AuthConfigHttpBasicAuthConfig(_messages.Message):
    """Config for HTTP Basic Authentication.

  Fields:
    credentialSecret: Required. The name of the SecretManager secret version
      resource storing the base64 encoded credentials. Format:
      `projects/{project}/secrets/{secrete}/versions/{version}` - If
      specified, the `secretmanager.versions.access` permission should be
      granted to Vertex AI Extension Service Agent
      (https://cloud.google.com/vertex-ai/docs/general/access-control#service-
      agents) on the specified resource.
  """
    credentialSecret = _messages.StringField(1)