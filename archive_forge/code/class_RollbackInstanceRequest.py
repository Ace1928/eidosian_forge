from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackInstanceRequest(_messages.Message):
    """Request for rollbacking a notebook instance

  Fields:
    revisionId: Required. Output only. Revision Id
    targetSnapshot: Required. The snapshot for rollback. Example:
      "projects/test-project/global/snapshots/krwlzipynril".
  """
    revisionId = _messages.StringField(1)
    targetSnapshot = _messages.StringField(2)