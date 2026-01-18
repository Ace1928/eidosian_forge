from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeIgnoredMembership(_messages.Message):
    """IgnoredMembership represents a membership ignored by the feature. A
  membership can be ignored because it was manually upgraded to a newer
  version than RC default.

  Fields:
    ignoredTime: Time when the membership was first set to ignored.
    reason: Reason why the membership is ignored.
  """
    ignoredTime = _messages.StringField(1)
    reason = _messages.StringField(2)