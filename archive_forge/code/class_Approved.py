from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Approved(_messages.Message):
    """An event representing that the Grant was approved.

  Fields:
    actor: Output only. Username of the user who approved the grant.
    reason: Output only. The reason provided by the approver for approving the
      Grant.
  """
    actor = _messages.StringField(1)
    reason = _messages.StringField(2)