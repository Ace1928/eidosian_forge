from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstanceMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateInstance.

  Enums:
    ExpectedFulfillmentPeriodValueValuesEnum: The expected fulfillment period
      of this update operation.

  Fields:
    cancelTime: The time at which this operation was cancelled. If set, this
      operation is in the process of undoing itself (which is guaranteed to
      succeed) and cannot be cancelled again.
    endTime: The time at which this operation failed or was completed
      successfully.
    expectedFulfillmentPeriod: The expected fulfillment period of this update
      operation.
    instance: The desired end state of the update.
    startTime: The time at which UpdateInstance request was received.
  """

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
    cancelTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    expectedFulfillmentPeriod = _messages.EnumField('ExpectedFulfillmentPeriodValueValuesEnum', 3)
    instance = _messages.MessageField('Instance', 4)
    startTime = _messages.StringField(5)