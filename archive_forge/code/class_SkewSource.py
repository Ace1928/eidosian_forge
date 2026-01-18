from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkewSource(_messages.Message):
    """Details about source stages which produce skewed data.

  Fields:
    stageId: Output only. Stage id of the skew source stage.
  """
    stageId = _messages.IntegerField(1)