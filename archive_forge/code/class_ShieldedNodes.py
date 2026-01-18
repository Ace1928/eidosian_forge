from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedNodes(_messages.Message):
    """Configuration of Shielded Nodes feature.

  Fields:
    enabled: Whether Shielded Nodes features are enabled on all nodes in this
      cluster.
  """
    enabled = _messages.BooleanField(1)