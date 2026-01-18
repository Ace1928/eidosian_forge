from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NotebookReservationAffinity(_messages.Message):
    """Notebook Reservation Affinity for consuming Zonal reservation.

  Enums:
    ConsumeReservationTypeValueValuesEnum: Required. Specifies the type of
      reservation from which this instance can consume resources:
      RESERVATION_ANY (default), RESERVATION_SPECIFIC, or RESERVATION_NONE.
      See Consuming reserved instances for examples.

  Fields:
    consumeReservationType: Required. Specifies the type of reservation from
      which this instance can consume resources: RESERVATION_ANY (default),
      RESERVATION_SPECIFIC, or RESERVATION_NONE. See Consuming reserved
      instances for examples.
    key: Optional. Corresponds to the label key of a reservation resource. To
      target a RESERVATION_SPECIFIC by name, use
      compute.googleapis.com/reservation-name as the key and specify the name
      of your reservation as its value.
    values: Optional. Corresponds to the label values of a reservation
      resource. This must be the full path name of Reservation.
  """

    class ConsumeReservationTypeValueValuesEnum(_messages.Enum):
        """Required. Specifies the type of reservation from which this instance
    can consume resources: RESERVATION_ANY (default), RESERVATION_SPECIFIC, or
    RESERVATION_NONE. See Consuming reserved instances for examples.

    Values:
      RESERVATION_AFFINITY_TYPE_UNSPECIFIED: Default type.
      RESERVATION_NONE: Do not consume from any allocated capacity.
      RESERVATION_ANY: Consume any reservation available.
      RESERVATION_SPECIFIC: Must consume from a specific reservation. Must
        specify key value fields for specifying the reservations.
    """
        RESERVATION_AFFINITY_TYPE_UNSPECIFIED = 0
        RESERVATION_NONE = 1
        RESERVATION_ANY = 2
        RESERVATION_SPECIFIC = 3
    consumeReservationType = _messages.EnumField('ConsumeReservationTypeValueValuesEnum', 1)
    key = _messages.StringField(2)
    values = _messages.StringField(3, repeated=True)