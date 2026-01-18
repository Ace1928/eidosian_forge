from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeGKEUpgradeFeatureState(_messages.Message):
    """GKEUpgradeFeatureState contains feature states for GKE clusters in the
  scope.

  Fields:
    conditions: Current conditions of the feature.
    upgradeState: Upgrade state. It will eventually replace `state`.
  """
    conditions = _messages.MessageField('ClusterUpgradeGKEUpgradeFeatureCondition', 1, repeated=True)
    upgradeState = _messages.MessageField('ClusterUpgradeGKEUpgradeState', 2, repeated=True)