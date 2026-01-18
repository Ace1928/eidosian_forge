from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdBreak(_messages.Message):
    """Ad break.

  Fields:
    startTimeOffset: Start time in seconds for the ad break, relative to the
      output file timeline. The default is `0s`.
  """
    startTimeOffset = _messages.StringField(1)