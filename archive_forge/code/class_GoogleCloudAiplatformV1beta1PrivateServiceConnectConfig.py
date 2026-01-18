from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PrivateServiceConnectConfig(_messages.Message):
    """Represents configuration for private service connect.

  Fields:
    enablePrivateServiceConnect: Required. If true, expose the IndexEndpoint
      via private service connect.
    projectAllowlist: A list of Projects from which the forwarding rule will
      target the service attachment.
  """
    enablePrivateServiceConnect = _messages.BooleanField(1)
    projectAllowlist = _messages.StringField(2, repeated=True)