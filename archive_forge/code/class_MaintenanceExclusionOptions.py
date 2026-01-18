from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceExclusionOptions(_messages.Message):
    """Represents the Maintenance exclusion option.

  Enums:
    ScopeValueValuesEnum: Scope specifies the upgrade scope which upgrades are
      blocked by the exclusion.

  Fields:
    scope: Scope specifies the upgrade scope which upgrades are blocked by the
      exclusion.
  """

    class ScopeValueValuesEnum(_messages.Enum):
        """Scope specifies the upgrade scope which upgrades are blocked by the
    exclusion.

    Values:
      NO_UPGRADES: NO_UPGRADES excludes all upgrades, including patch upgrades
        and minor upgrades across control planes and nodes. This is the
        default exclusion behavior.
      NO_MINOR_UPGRADES: NO_MINOR_UPGRADES excludes all minor upgrades for the
        cluster, only patches are allowed.
      NO_MINOR_OR_NODE_UPGRADES: NO_MINOR_OR_NODE_UPGRADES excludes all minor
        upgrades for the cluster, and also exclude all node pool upgrades.
        Only control plane patches are allowed.
    """
        NO_UPGRADES = 0
        NO_MINOR_UPGRADES = 1
        NO_MINOR_OR_NODE_UPGRADES = 2
    scope = _messages.EnumField('ScopeValueValuesEnum', 1)