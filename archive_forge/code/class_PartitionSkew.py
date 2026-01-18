from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionSkew(_messages.Message):
    """Partition skew detailed information.

  Fields:
    skewSources: Output only. Source stages which produce skewed data.
  """
    skewSources = _messages.MessageField('SkewSource', 1, repeated=True)