from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeGKEUpgrade(_messages.Message):
    """GKEUpgrade represents a GKE provided upgrade, e.g., control plane
  upgrade.

  Fields:
    name: Name of the upgrade, e.g., "k8s_control_plane". It should be a valid
      upgrade name. It must not exceet 99 characters.
    version: Version of the upgrade, e.g., "1.22.1-gke.100". It should be a
      valid version. It must not exceet 99 characters.
  """
    name = _messages.StringField(1)
    version = _messages.StringField(2)