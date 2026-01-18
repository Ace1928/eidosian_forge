from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollingSettings(_messages.Message):
    """Settings for rolling update.

  Fields:
    maxSurgePercentage: Percentage of the maximum number of nodes that can be
      created beyond the current size of the node pool during the upgrade
      process. The range of this field should be [0, 100].
    maxUnavailablePercentage: Percentage of the maximum number of nodes that
      can be unavailable during during the upgrade process.
  """
    maxSurgePercentage = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxUnavailablePercentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)