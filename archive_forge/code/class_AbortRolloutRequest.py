from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AbortRolloutRequest(_messages.Message):
    """Message for aborting a rollout.

  Fields:
    reason: Optional. Reason for aborting.
  """
    reason = _messages.StringField(1)