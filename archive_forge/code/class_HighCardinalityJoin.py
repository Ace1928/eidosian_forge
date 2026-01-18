from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HighCardinalityJoin(_messages.Message):
    """High cardinality join detailed information.

  Fields:
    leftRows: Output only. Count of left input rows.
    outputRows: Output only. Count of the output rows.
    rightRows: Output only. Count of right input rows.
    stepIndex: Output only. The index of the join operator in the
      ExplainQueryStep lists.
  """
    leftRows = _messages.IntegerField(1)
    outputRows = _messages.IntegerField(2)
    rightRows = _messages.IntegerField(3)
    stepIndex = _messages.IntegerField(4, variant=_messages.Variant.INT32)