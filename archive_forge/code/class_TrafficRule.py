from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TrafficRule(_messages.Message):
    """Network emulation parameters.

  Fields:
    bandwidth: Bandwidth in kbits/second.
    burst: Burst size in kbits.
    delay: Packet delay, must be >= 0.
    packetDuplicationRatio: Packet duplication ratio (0.0 - 1.0).
    packetLossRatio: Packet loss ratio (0.0 - 1.0).
  """
    bandwidth = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    burst = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    delay = _messages.StringField(3)
    packetDuplicationRatio = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    packetLossRatio = _messages.FloatField(5, variant=_messages.Variant.FLOAT)