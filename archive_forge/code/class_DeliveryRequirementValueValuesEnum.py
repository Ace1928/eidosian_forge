from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeliveryRequirementValueValuesEnum(_messages.Enum):
    """The DeliveryRequirement for this subscription.

    Values:
      DELIVERY_REQUIREMENT_UNSPECIFIED: Default value. This value is unused.
      DELIVER_IMMEDIATELY: The server does not wait for a published message to
        be successfully written to storage before delivering it to
        subscribers.
      DELIVER_AFTER_STORED: The server will not deliver a published message to
        subscribers until the message has been successfully written to
        storage. This will result in higher end-to-end latency, but consistent
        delivery.
    """
    DELIVERY_REQUIREMENT_UNSPECIFIED = 0
    DELIVER_IMMEDIATELY = 1
    DELIVER_AFTER_STORED = 2