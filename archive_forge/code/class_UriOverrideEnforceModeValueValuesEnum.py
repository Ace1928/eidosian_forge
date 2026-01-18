from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UriOverrideEnforceModeValueValuesEnum(_messages.Enum):
    """URI Override Enforce Mode When specified, determines the Target
    UriOverride mode. If not specified, it defaults to ALWAYS.

    Values:
      URI_OVERRIDE_ENFORCE_MODE_UNSPECIFIED: UriOverrideEnforceMode
        Unspecified. Defaults to ALWAYS.
      IF_NOT_EXISTS: In the IF_NOT_EXISTS mode, queue-level configuration is
        only applied where task-level configuration does not exist.
      ALWAYS: In the ALWAYS mode, queue-level configuration overrides all
        task-level configuration
    """
    URI_OVERRIDE_ENFORCE_MODE_UNSPECIFIED = 0
    IF_NOT_EXISTS = 1
    ALWAYS = 2