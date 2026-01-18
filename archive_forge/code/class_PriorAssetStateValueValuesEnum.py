from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PriorAssetStateValueValuesEnum(_messages.Enum):
    """State of prior_asset.

    Values:
      PRIOR_ASSET_STATE_UNSPECIFIED: prior_asset is not applicable for the
        current asset.
      PRESENT: prior_asset is populated correctly.
      INVALID: Failed to set prior_asset.
      DOES_NOT_EXIST: Current asset is the first known state.
      DELETED: prior_asset is a deletion.
    """
    PRIOR_ASSET_STATE_UNSPECIFIED = 0
    PRESENT = 1
    INVALID = 2
    DOES_NOT_EXIST = 3
    DELETED = 4