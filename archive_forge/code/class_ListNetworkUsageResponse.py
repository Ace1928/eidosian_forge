from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNetworkUsageResponse(_messages.Message):
    """Response with Networks with IPs

  Fields:
    networks: Networks with IPs.
  """
    networks = _messages.MessageField('NetworkUsage', 1, repeated=True)