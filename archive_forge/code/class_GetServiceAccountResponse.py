from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetServiceAccountResponse(_messages.Message):
    """Response object of GetServiceAccount

  Fields:
    email: The service account email address.
    kind: The resource type of the response.
  """
    email = _messages.StringField(1)
    kind = _messages.StringField(2, default='bigquery#getServiceAccountResponse')