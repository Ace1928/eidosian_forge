from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscaledRolloutPolicy(_messages.Message):
    """Autoscaled rollout policy uses cluster autoscaler during blue-green
  upgrades to scale both the green and blue pools.
  """