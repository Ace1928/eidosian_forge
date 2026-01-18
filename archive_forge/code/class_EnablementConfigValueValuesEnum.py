from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnablementConfigValueValuesEnum(_messages.Enum):
    """Optional. Config for whether this repository has vulnerability
    scanning disabled.

    Values:
      ENABLEMENT_CONFIG_UNSPECIFIED: Unspecified config was not set. This will
        be interpreted as DISABLED. On Repository creation, UNSPECIFIED
        vulnerability scanning will be defaulted to INHERITED.
      INHERITED: Inherited indicates the repository is allowed for
        vulnerability scanning, however the actual state will be inherited
        from the API enablement state.
      DISABLED: Disabled indicates the repository will not perform
        vulnerability scanning.
    """
    ENABLEMENT_CONFIG_UNSPECIFIED = 0
    INHERITED = 1
    DISABLED = 2