from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationStatus(_messages.Message):
    """[Output only] Represents status related to the future reservation.

  Enums:
    AmendmentStatusValueValuesEnum: [Output Only] The current status of the
      requested amendment.
    ProcurementStatusValueValuesEnum: Current state of this Future Reservation

  Fields:
    amendmentStatus: [Output Only] The current status of the requested
      amendment.
    autoCreatedReservations: Fully qualified urls of the automatically created
      reservations at start_time.
    fulfilledCount: This count indicates the fulfilled capacity so far. This
      is set during "PROVISIONING" state. This count also includes capacity
      delivered as part of existing matching reservations.
    lastKnownGoodState: [Output Only] This field represents the future
      reservation before an amendment was requested. If the amendment is
      declined, the Future Reservation will be reverted to the last known good
      state. The last known good state is not set when updating a future
      reservation whose Procurement Status is DRAFTING.
    lockTime: Time when Future Reservation would become LOCKED, after which no
      modifications to Future Reservation will be allowed. Applicable only
      after the Future Reservation is in the APPROVED state. The lock_time is
      an RFC3339 string. The procurement_status will transition to PROCURING
      state at this time.
    procurementStatus: Current state of this Future Reservation
    specificSkuProperties: A FutureReservationStatusSpecificSKUProperties
      attribute.
  """

    class AmendmentStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The current status of the requested amendment.

    Values:
      AMENDMENT_APPROVED: The requested amendment to the Future Resevation has
        been approved and applied by GCP.
      AMENDMENT_DECLINED: The requested amendment to the Future Reservation
        has been declined by GCP and the original state was restored.
      AMENDMENT_IN_REVIEW: The requested amendment to the Future Reservation
        is currently being reviewd by GCP.
      AMENDMENT_STATUS_UNSPECIFIED: <no description>
    """
        AMENDMENT_APPROVED = 0
        AMENDMENT_DECLINED = 1
        AMENDMENT_IN_REVIEW = 2
        AMENDMENT_STATUS_UNSPECIFIED = 3

    class ProcurementStatusValueValuesEnum(_messages.Enum):
        """Current state of this Future Reservation

    Values:
      APPROVED: Future reservation is approved by GCP.
      CANCELLED: Future reservation is cancelled by the customer.
      COMMITTED: Future reservation is committed by the customer.
      DECLINED: Future reservation is rejected by GCP.
      DRAFTING: Related status for PlanningStatus.Draft. Transitions to
        PENDING_APPROVAL upon user submitting FR.
      FAILED: Future reservation failed. No additional reservations were
        provided.
      FAILED_PARTIALLY_FULFILLED: Future reservation is partially fulfilled.
        Additional reservations were provided but did not reach total_count
        reserved instance slots.
      FULFILLED: Future reservation is fulfilled completely.
      PENDING_AMENDMENT_APPROVAL: An Amendment to the Future Reservation has
        been requested. If the Amendment is declined, the Future Reservation
        will be restored to the last known good state.
      PENDING_APPROVAL: Future reservation is pending approval by GCP.
      PROCUREMENT_STATUS_UNSPECIFIED: <no description>
      PROCURING: Future reservation is being procured by GCP. Beyond this
        point, Future reservation is locked and no further modifications are
        allowed.
      PROVISIONING: Future reservation capacity is being provisioned. This
        state will be entered after start_time, while reservations are being
        created to provide total_count reserved instance slots. This state
        will not persist past start_time + 24h.
    """
        APPROVED = 0
        CANCELLED = 1
        COMMITTED = 2
        DECLINED = 3
        DRAFTING = 4
        FAILED = 5
        FAILED_PARTIALLY_FULFILLED = 6
        FULFILLED = 7
        PENDING_AMENDMENT_APPROVAL = 8
        PENDING_APPROVAL = 9
        PROCUREMENT_STATUS_UNSPECIFIED = 10
        PROCURING = 11
        PROVISIONING = 12
    amendmentStatus = _messages.EnumField('AmendmentStatusValueValuesEnum', 1)
    autoCreatedReservations = _messages.StringField(2, repeated=True)
    fulfilledCount = _messages.IntegerField(3)
    lastKnownGoodState = _messages.MessageField('FutureReservationStatusLastKnownGoodState', 4)
    lockTime = _messages.StringField(5)
    procurementStatus = _messages.EnumField('ProcurementStatusValueValuesEnum', 6)
    specificSkuProperties = _messages.MessageField('FutureReservationStatusSpecificSKUProperties', 7)