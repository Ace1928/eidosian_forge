from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpectedFulfillmentPeriodValueValuesEnum(_messages.Enum):
    """The expected fulfillment period of this update operation.

    Values:
      FULFILLMENT_PERIOD_UNSPECIFIED: Not specified.
      FULFILLMENT_PERIOD_NORMAL: Normal fulfillment period. The operation is
        expected to complete within minutes.
      FULFILLMENT_PERIOD_EXTENDED: Extended fulfillment period. It can take up
        to an hour for the operation to complete.
    """
    FULFILLMENT_PERIOD_UNSPECIFIED = 0
    FULFILLMENT_PERIOD_NORMAL = 1
    FULFILLMENT_PERIOD_EXTENDED = 2