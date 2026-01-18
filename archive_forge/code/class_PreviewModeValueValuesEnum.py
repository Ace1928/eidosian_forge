from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreviewModeValueValuesEnum(_messages.Enum):
    """Optional. Current mode of preview.

    Values:
      PREVIEW_MODE_UNSPECIFIED: Unspecified policy, default mode will be used.
      DEFAULT: DEFAULT mode generates an execution plan for reconciling
        current resource state into expected resource state.
      DELETE: DELETE mode generates as execution plan for destroying current
        resources.
    """
    PREVIEW_MODE_UNSPECIFIED = 0
    DEFAULT = 1
    DELETE = 2