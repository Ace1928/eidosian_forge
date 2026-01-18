from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkPolicyConfig(_messages.Message):
    """Configuration for NetworkPolicy. This only tracks whether the addon is
  enabled or not on the Master, it does not track whether network policy is
  enabled for the nodes.

  Fields:
    disabled: Whether NetworkPolicy is enabled for this cluster.
  """
    disabled = _messages.BooleanField(1)