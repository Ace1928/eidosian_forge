from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetPackageInfo(_messages.Message):
    """FleetPackageInfo represents the status of resource bundle rollout across
  the target clusters.

  Enums:
    StateValueValuesEnum: Optional. Output only. The current state of a
      FleetPackage.

  Fields:
    activeRollout: Optional. The active rollout, if any. Format is `projects/{
      project}/locations/{location}/fleetPackages/{fleet_package}/rollouts/{ro
      llout}`.
    lastCompletedRollout: Optional. The last completed rollout, if any. Format
      is `projects/{project}/locations/{location}/fleetPackages/{fleet_package
      }/rollouts/{rollout}`.
    state: Optional. Output only. The current state of a FleetPackage.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Optional. Output only. The current state of a FleetPackage.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: FleetPackage is active.
      SUSPENDED: FleetPackage is suspended.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        SUSPENDED = 2
    activeRollout = _messages.StringField(1)
    lastCompletedRollout = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)