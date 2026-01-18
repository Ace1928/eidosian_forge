from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsSetManagedProtectionTierRequest(_messages.Message):
    """A ProjectsSetManagedProtectionTierRequest object.

  Enums:
    ManagedProtectionTierValueValuesEnum: Managed protection tier to be set.

  Fields:
    managedProtectionTier: Managed protection tier to be set.
  """

    class ManagedProtectionTierValueValuesEnum(_messages.Enum):
        """Managed protection tier to be set.

    Values:
      CAMP_PLUS_ANNUAL: Plus tier protection annual.
      CAMP_PLUS_PAYGO: Plus tier protection monthly.
      CA_STANDARD: Standard protection.
    """
        CAMP_PLUS_ANNUAL = 0
        CAMP_PLUS_PAYGO = 1
        CA_STANDARD = 2
    managedProtectionTier = _messages.EnumField('ManagedProtectionTierValueValuesEnum', 1)