from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetStatus(_messages.Message):
    """Aggregated status of the underlying assets of a lake or zone.

  Fields:
    activeAssets: Number of active assets.
    securityPolicyApplyingAssets: Number of assets that are in process of
      updating the security policy on attached resources.
    updateTime: Last update time of the status.
  """
    activeAssets = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    securityPolicyApplyingAssets = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    updateTime = _messages.StringField(3)