from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyResourceStatusInstanceSchedulePolicyStatus(_messages.Message):
    """A ResourcePolicyResourceStatusInstanceSchedulePolicyStatus object.

  Fields:
    lastRunStartTime: [Output Only] The last time the schedule successfully
      ran. The timestamp is an RFC3339 string.
    nextRunStartTime: [Output Only] The next time the schedule is planned to
      run. The actual time might be slightly different. The timestamp is an
      RFC3339 string.
  """
    lastRunStartTime = _messages.StringField(1)
    nextRunStartTime = _messages.StringField(2)