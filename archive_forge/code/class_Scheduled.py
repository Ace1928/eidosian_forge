from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Scheduled(_messages.Message):
    """An event representing that the Grant has been scheduled to be activated
  later.

  Fields:
    scheduledActivationTime: Output only. The time at which the access will be
      granted.
  """
    scheduledActivationTime = _messages.StringField(1)