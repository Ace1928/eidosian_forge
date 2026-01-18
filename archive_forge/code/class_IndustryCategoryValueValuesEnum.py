from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndustryCategoryValueValuesEnum(_messages.Enum):
    """The group of relevant businesses where this infoType is commonly used

    Values:
      INDUSTRY_UNSPECIFIED: Unused industry
      FINANCE: The infoType is typically used in the finance industry.
      HEALTH: The infoType is typically used in the health industry.
      TELECOMMUNICATIONS: The infoType is typically used in the
        telecommunications industry.
    """
    INDUSTRY_UNSPECIFIED = 0
    FINANCE = 1
    HEALTH = 2
    TELECOMMUNICATIONS = 3