from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ServiceIssuersMapping(_messages.Message):
    """A GoogleCloudApigeeV1ServiceIssuersMapping object.

  Fields:
    emailIds: List of trusted issuer email ids.
    service: String indicating the Apigee service name.
  """
    emailIds = _messages.StringField(1, repeated=True)
    service = _messages.StringField(2)