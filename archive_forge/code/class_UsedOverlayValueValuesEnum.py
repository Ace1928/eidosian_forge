from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsedOverlayValueValuesEnum(_messages.Enum):
    """Indicates whether overlay was used.

    Values:
      OVERLAY_UNSPECIFIED: Indicates that whether or not overlay was used is
        unspecified. This applies, for example, to non-CommandTask actions.
      OVERLAY_ENABLED: Indicates that overlay was used for a CommandTask
        action.
      OVERLAY_DISABLED: Indicates that overlay was not used for a CommandTask
        action.
    """
    OVERLAY_UNSPECIFIED = 0
    OVERLAY_ENABLED = 1
    OVERLAY_DISABLED = 2