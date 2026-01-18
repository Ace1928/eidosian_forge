from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpConfig(_messages.Message):
    """Defines IP configuration where this Certificate Map is serving.

  Fields:
    ipAddress: Output only. An external IP address.
    ports: Output only. Ports.
  """
    ipAddress = _messages.StringField(1)
    ports = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.UINT32)