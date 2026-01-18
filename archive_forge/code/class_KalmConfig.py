from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KalmConfig(_messages.Message):
    """Configuration options for the KALM addon.

  Fields:
    enabled: Whether KALM is enabled for this cluster.
  """
    enabled = _messages.BooleanField(1)