from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureResourceState(_messages.Message):
    """FeatureResourceState describes the state of a Feature *resource* in the
  GkeHub API. See `FeatureState` for the "running state" of the Feature in the
  Hub and across Memberships.

  Enums:
    StateValueValuesEnum: The current state of the Feature resource in the Hub
      API.

  Fields:
    state: The current state of the Feature resource in the Hub API.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the Feature resource in the Hub API.

    Values:
      STATE_UNSPECIFIED: State is unknown or not set.
      ENABLING: The Feature is being enabled, and the Feature resource is
        being created. Once complete, the corresponding Feature will be
        enabled in this Hub.
      ACTIVE: The Feature is enabled in this Hub, and the Feature resource is
        fully available.
      DISABLING: The Feature is being disabled in this Hub, and the Feature
        resource is being deleted.
      UPDATING: The Feature resource is being updated.
      SERVICE_UPDATING: The Feature resource is being updated by the Hub
        Service.
    """
        STATE_UNSPECIFIED = 0
        ENABLING = 1
        ACTIVE = 2
        DISABLING = 3
        UPDATING = 4
        SERVICE_UPDATING = 5
    state = _messages.EnumField('StateValueValuesEnum', 1)