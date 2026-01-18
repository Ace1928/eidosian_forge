from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest(_messages.Message):
    """Request message for ConsumerProcurementService.CancelOrder.

  Enums:
    CancellationPolicyValueValuesEnum: Cancellation policy of this request.

  Fields:
    cancellationPolicy: Cancellation policy of this request.
    etag: The weak etag, which can be optionally populated, of the order that
      this cancel request is based on. Validation checking will only happen if
      the invoker supplies this field.
  """

    class CancellationPolicyValueValuesEnum(_messages.Enum):
        """Cancellation policy of this request.

    Values:
      CANCELLATION_POLICY_UNSPECIFIED: If unspecified, cancellation will try
        to cancel the order, if order cannot be immediately cancelled, auto
        renewal will be turned off. However, caller should avoid using the
        value as it will yield a non-deterministic result. This is still
        supported mainly to maintain existing integrated usages and ensure
        backwards compatibility.
      CANCELLATION_POLICY_CANCEL_IMMEDIATELY: Request will cancel the whole
        order immediately, if order cannot be immediately cancelled, the
        request will fail.
      CANCELLATION_POLICY_CANCEL_AT_TERM_END: Request will cancel the auto
        renewal, if order is not subscription based, the request will fail.
    """
        CANCELLATION_POLICY_UNSPECIFIED = 0
        CANCELLATION_POLICY_CANCEL_IMMEDIATELY = 1
        CANCELLATION_POLICY_CANCEL_AT_TERM_END = 2
    cancellationPolicy = _messages.EnumField('CancellationPolicyValueValuesEnum', 1)
    etag = _messages.StringField(2)