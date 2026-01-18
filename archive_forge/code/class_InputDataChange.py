from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InputDataChange(_messages.Message):
    """Details about the input data change insight.

  Fields:
    recordsReadDiffPercentage: Output only. Records read difference percentage
      compared to a previous run.
  """
    recordsReadDiffPercentage = _messages.FloatField(1, variant=_messages.Variant.FLOAT)