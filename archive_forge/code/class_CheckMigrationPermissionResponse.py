from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckMigrationPermissionResponse(_messages.Message):
    """CheckMigrationPermissionResponse is the response message for
  CheckMigrationPermission method.

  Enums:
    StateValueValuesEnum: The state of DomainMigration.

  Fields:
    onpremDomains: The state of SID filtering of all the domains which has
      trust established.
    state: The state of DomainMigration.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of DomainMigration.

    Values:
      STATE_UNSPECIFIED: DomainMigration is in unspecified state.
      DISABLED: Domain Migration is Disabled.
      ENABLED: Domain Migration is Enabled.
      NEEDS_MAINTENANCE: Domain Migration is not in valid state.
    """
        STATE_UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2
        NEEDS_MAINTENANCE = 3
    onpremDomains = _messages.MessageField('OnPremDomainSIDDetails', 1, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 2)