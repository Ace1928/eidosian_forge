from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Denied(_messages.Message):
    """An event representing that the Grant was denied.

  Fields:
    actor: Output only. Username of the user who denied the grant.
    reason: Output only. The reason provided by the approver for denying the
      Grant.
  """
    actor = _messages.StringField(1)
    reason = _messages.StringField(2)