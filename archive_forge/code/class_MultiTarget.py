from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiTarget(_messages.Message):
    """Information specifying a multiTarget.

  Fields:
    targetIds: Required. The target_ids of this multiTarget.
  """
    targetIds = _messages.StringField(1, repeated=True)