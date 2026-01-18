from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TcpCheck(_messages.Message):
    """Information required for a TCP Uptime check request.

  Fields:
    pingConfig: Contains information needed to add pings to a TCP check.
    port: The TCP port on the server against which to run the check. Will be
      combined with host (specified within the monitored_resource) to
      construct the full URL. Required.
  """
    pingConfig = _messages.MessageField('PingConfig', 1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)