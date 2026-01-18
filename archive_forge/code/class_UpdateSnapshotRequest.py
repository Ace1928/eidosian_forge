from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UpdateSnapshotRequest(_messages.Message):
    """Request for the UpdateSnapshot method.

  Fields:
    snapshot: Required. The updated snapshot object.
    updateMask: Required. Indicates which fields in the provided snapshot to
      update. Must be specified and non-empty.
  """
    snapshot = _messages.MessageField('Snapshot', 1)
    updateMask = _messages.StringField(2)