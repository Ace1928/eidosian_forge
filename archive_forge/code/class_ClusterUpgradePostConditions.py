from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradePostConditions(_messages.Message):
    """Post conditional checks after an upgrade has been applied on all
  eligible clusters.

  Fields:
    soaking: Required. Amount of time to "soak" after a rollout has been
      finished before marking it COMPLETE. Cannot exceed 30 days. Required.
  """
    soaking = _messages.StringField(1)