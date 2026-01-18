from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutStrategy(_messages.Message):
    """RolloutStrategy defines different ways to rollout a resource bundle
  across a set of clusters.

  Fields:
    allAtOnce: AllAtOnceStrategy causes all clusters to be updated
      concurrently.
    rolling: RollingStrategy causes a specified number of clusters to be
      updated concurrently until all clusters are updated.
  """
    allAtOnce = _messages.MessageField('AllAtOnceStrategy', 1)
    rolling = _messages.MessageField('RollingStrategy', 2)