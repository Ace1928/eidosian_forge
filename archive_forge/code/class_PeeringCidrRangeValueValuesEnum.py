from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeringCidrRangeValueValuesEnum(_messages.Enum):
    """Optional. Size of the CIDR block range that will be reserved by the
    instance. PAID organizations support `SLASH_16` to `SLASH_20` and defaults
    to `SLASH_16`. Evaluation organizations support only `SLASH_23`.

    Values:
      CIDR_RANGE_UNSPECIFIED: Range not specified.
      SLASH_16: `/16` CIDR range.
      SLASH_17: `/17` CIDR range.
      SLASH_18: `/18` CIDR range.
      SLASH_19: `/19` CIDR range.
      SLASH_20: `/20` CIDR range.
      SLASH_22: `/22` CIDR range. Supported for evaluation only.
      SLASH_23: `/23` CIDR range. Supported for evaluation only.
    """
    CIDR_RANGE_UNSPECIFIED = 0
    SLASH_16 = 1
    SLASH_17 = 2
    SLASH_18 = 3
    SLASH_19 = 4
    SLASH_20 = 5
    SLASH_22 = 6
    SLASH_23 = 7