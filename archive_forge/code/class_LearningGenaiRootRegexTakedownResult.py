from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRegexTakedownResult(_messages.Message):
    """A LearningGenaiRootRegexTakedownResult object.

  Fields:
    allowed: False when query or response should be taken down due to match
      with a blocked regex, true otherwise.
    takedownRegex: Regex used to decide that query or response should be taken
      down. Empty when query or response is kept.
  """
    allowed = _messages.BooleanField(1)
    takedownRegex = _messages.StringField(2)