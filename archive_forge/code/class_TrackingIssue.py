from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TrackingIssue(_messages.Message):
    """Information related to tracking the progress on resolving the error.

  Fields:
    url: A URL pointing to a related entry in an issue tracking system.
      Example: `https://github.com/user/project/issues/4`
  """
    url = _messages.StringField(1)