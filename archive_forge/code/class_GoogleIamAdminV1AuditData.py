from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1AuditData(_messages.Message):
    """Audit log information specific to Cloud IAM admin APIs. This message is
  serialized as an `Any` type in the `ServiceData` message of an `AuditLog`
  message.

  Fields:
    permissionDelta: The permission_delta when when creating or updating a
      Role.
  """
    permissionDelta = _messages.MessageField('GoogleIamAdminV1AuditDataPermissionDelta', 1)