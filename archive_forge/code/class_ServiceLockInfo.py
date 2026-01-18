from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLockInfo(_messages.Message):
    """ServiceLockInfo represents the details of a lock taken by the service on
  a Backup resource.

  Fields:
    operation: Output only. The name of the operation that created this lock.
      The lock will automatically be released when the operation completes.
  """
    operation = _messages.StringField(1)