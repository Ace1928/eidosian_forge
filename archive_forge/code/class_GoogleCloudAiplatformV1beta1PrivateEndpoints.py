from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PrivateEndpoints(_messages.Message):
    """PrivateEndpoints proto is used to provide paths for users to send
  requests privately. To send request via private service access, use
  predict_http_uri, explain_http_uri or health_http_uri. To send request via
  private service connect, use service_attachment.

  Fields:
    explainHttpUri: Output only. Http(s) path to send explain requests.
    healthHttpUri: Output only. Http(s) path to send health check requests.
    predictHttpUri: Output only. Http(s) path to send prediction requests.
    serviceAttachment: Output only. The name of the service attachment
      resource. Populated if private service connect is enabled.
  """
    explainHttpUri = _messages.StringField(1)
    healthHttpUri = _messages.StringField(2)
    predictHttpUri = _messages.StringField(3)
    serviceAttachment = _messages.StringField(4)