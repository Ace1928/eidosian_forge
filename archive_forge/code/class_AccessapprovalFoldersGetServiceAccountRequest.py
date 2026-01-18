from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalFoldersGetServiceAccountRequest(_messages.Message):
    """A AccessapprovalFoldersGetServiceAccountRequest object.

  Fields:
    name: Name of the AccessApprovalServiceAccount to retrieve.
  """
    name = _messages.StringField(1, required=True)