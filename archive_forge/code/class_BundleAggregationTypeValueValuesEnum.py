from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BundleAggregationTypeValueValuesEnum(_messages.Enum):
    """The aggregation type of the bundle interface.

    Values:
      BUNDLE_AGGREGATION_TYPE_LACP: LACP is enabled.
      BUNDLE_AGGREGATION_TYPE_STATIC: LACP is disabled.
    """
    BUNDLE_AGGREGATION_TYPE_LACP = 0
    BUNDLE_AGGREGATION_TYPE_STATIC = 1