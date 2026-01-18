from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootCodeyTruncatorMetadata(_messages.Message):
    """Metadata describing what was truncated at each checkpoint.

  Fields:
    cutoffIndex: Index of the current sample that trims off truncated text.
    truncatedText: Text that was truncated at a specific checkpoint.
  """
    cutoffIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    truncatedText = _messages.StringField(2)