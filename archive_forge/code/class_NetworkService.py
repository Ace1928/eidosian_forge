from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkService(_messages.Message):
    """Represents a network service that is managed by a `NetworkPolicy`
  resource. A network service provides a way to control an aspect of external
  access to VMware workloads. For example, whether the VMware workloads in the
  private clouds governed by a network policy can access or be accessed from
  the internet.

  Enums:
    StateValueValuesEnum: Output only. State of the service. New values may be
      added to this enum when appropriate.

  Fields:
    enabled: True if the service is enabled; false otherwise.
    state: Output only. State of the service. New values may be added to this
      enum when appropriate.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the service. New values may be added to this
    enum when appropriate.

    Values:
      STATE_UNSPECIFIED: Unspecified service state. This is the default value.
      UNPROVISIONED: Service is not provisioned.
      RECONCILING: Service is in the process of being
        provisioned/deprovisioned.
      ACTIVE: Service is active.
    """
        STATE_UNSPECIFIED = 0
        UNPROVISIONED = 1
        RECONCILING = 2
        ACTIVE = 3
    enabled = _messages.BooleanField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)