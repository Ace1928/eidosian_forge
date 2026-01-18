from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsCacheConfig(_messages.Message):
    """Configuration for NodeLocal DNSCache

  Fields:
    enabled: Whether NodeLocal DNSCache is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)