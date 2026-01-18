from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdditionalPodRangesConfig(_messages.Message):
    """AdditionalPodRangesConfig is the configuration for additional pod
  secondary ranges supporting the ClusterUpdate message.

  Fields:
    podRangeInfo: Output only. [Output only] Information for additional pod
      range.
    podRangeNames: Name for pod secondary ipv4 range which has the actual
      range defined ahead.
  """
    podRangeInfo = _messages.MessageField('RangeInfo', 1, repeated=True)
    podRangeNames = _messages.StringField(2, repeated=True)