from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkuValueValuesEnum(_messages.Enum):
    """Required. SKU of subscription.

    Values:
      SKU_UNSPECIFIED: Default value. This value is unused.
      BCE_STANDARD_SKU: Represents BeyondCorp Standard SKU.
    """
    SKU_UNSPECIFIED = 0
    BCE_STANDARD_SKU = 1