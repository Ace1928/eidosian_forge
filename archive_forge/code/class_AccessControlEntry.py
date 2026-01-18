from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessControlEntry(_messages.Message):
    """AccessControlEntry is used to control who can do some operation.

  Fields:
    principals: Optional. Users who are being allowed for the operation. Each
      entry should be a valid v1 IAM Principal Identifier. Format for these is
      documented at: https://cloud.google.com/iam/docs/principal-
      identifiers#v1
  """
    principals = _messages.StringField(1, repeated=True)