from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FastSocket(_messages.Message):
    """Configuration of Fast Socket feature.

  Fields:
    enabled: Whether Fast Socket features are enabled in the node pool.
  """
    enabled = _messages.BooleanField(1)