from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnabledValueValuesEnum(_messages.Enum):
    """Optional. Provenance push mode.

    Values:
      ENABLED_UNSPECIFIED: Default to disabled (before AA regionalization),
        optimistic after
      REQUIRED: Provenance failures would fail the run
      OPTIMISTIC: GCB will attempt to push to artifact analaysis and build
        state would not be impacted by the push failures.
      DISABLED: Disable the provenance push entirely.
    """
    ENABLED_UNSPECIFIED = 0
    REQUIRED = 1
    OPTIMISTIC = 2
    DISABLED = 3