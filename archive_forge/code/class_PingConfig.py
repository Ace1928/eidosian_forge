from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PingConfig(_messages.Message):
    """Information involved in sending ICMP pings alongside public HTTP/TCP
  checks. For HTTP, the pings are performed for each part of the redirect
  chain.

  Fields:
    pingsCount: Number of ICMP pings. A maximum of 3 ICMP pings is currently
      supported.
  """
    pingsCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)