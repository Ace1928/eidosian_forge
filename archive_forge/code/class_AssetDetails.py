from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssetDetails(_messages.Message):
    """A AssetDetails object.

  Fields:
    asset: JSON string representing CAI asset. Format/representation may
      change, thus clients should not depend.
    assetType: Type of asset. See CAI asset type for GCP assets:
      https://cloud.google.com/asset-inventory/docs/supported-asset-types.
  """
    asset = _messages.StringField(1)
    assetType = _messages.StringField(2)