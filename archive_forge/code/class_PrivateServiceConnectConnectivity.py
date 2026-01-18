from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateServiceConnectConnectivity(_messages.Message):
    """[Private Service Connect
  connectivity](https://cloud.google.com/vpc/docs/private-service-
  connect#service-attachments)

  Fields:
    serviceAttachment: Required. A service attachment that exposes a database,
      and has the following format: projects/{project}/regions/{region}/servic
      eAttachments/{service_attachment_name}
  """
    serviceAttachment = _messages.StringField(1)