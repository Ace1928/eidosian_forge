from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Bgp(_messages.Message):
    """BGP information specific to this router.

  Fields:
    asn: Locally assigned BGP ASN.
    keepaliveIntervalInSeconds: The interval in seconds between BGP keepalive
      messages that are sent to the peer. Default is 20 with value between 20
      and 60.
  """
    asn = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    keepaliveIntervalInSeconds = _messages.IntegerField(2, variant=_messages.Variant.UINT32)