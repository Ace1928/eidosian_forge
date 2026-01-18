from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrderStateValueValuesEnum(_messages.Enum):
    """Output only. The state of the order.

    Values:
      ORDER_STATE_UNSPECIFIED: Sentinel value. Do not use.
      ORDER_STATE_ACTIVE: The order is active.
      ORDER_STATE_CANCELLED: The order is cancelled.
      ORDER_STATE_PENDING_CANCELLATION: The order is being cancelled either by
        the user or by the system. The order stays in this state, if any
        product in the order allows use of the underlying resource until the
        end of the current billing cycle. Once the billing cycle completes,
        the resource will transition to OrderState.ORDER_STATE_CANCELLED
        state.
    """
    ORDER_STATE_UNSPECIFIED = 0
    ORDER_STATE_ACTIVE = 1
    ORDER_STATE_CANCELLED = 2
    ORDER_STATE_PENDING_CANCELLATION = 3