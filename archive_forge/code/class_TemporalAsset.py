from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemporalAsset(_messages.Message):
    """An asset in Google Cloud and its temporal metadata, including the time
  window when it was observed and its status during that window.

  Enums:
    PriorAssetStateValueValuesEnum: State of prior_asset.

  Fields:
    asset: An asset in Google Cloud.
    deleted: Whether the asset has been deleted or not.
    priorAsset: Prior copy of the asset. Populated if prior_asset_state is
      PRESENT. Currently this is only set for responses in Real-Time Feed.
    priorAssetState: State of prior_asset.
    window: The time window when the asset data and state was observed.
  """

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
    asset = _messages.MessageField('Asset', 1)
    deleted = _messages.BooleanField(2)
    priorAsset = _messages.MessageField('Asset', 3)
    priorAssetState = _messages.EnumField('PriorAssetStateValueValuesEnum', 4)
    window = _messages.MessageField('TimeWindow', 5)