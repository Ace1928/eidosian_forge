from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1ReliabilityProjection(_messages.Message):
    """Contains information on the impact of a reliability recommendation.

  Enums:
    RisksValueListEntryValuesEnum:

  Messages:
    DetailsValue: Per-recommender projection.

  Fields:
    details: Per-recommender projection.
    risks: Reliability risks mitigated by this recommendation.
  """

    class RisksValueListEntryValuesEnum(_messages.Enum):
        """RisksValueListEntryValuesEnum enum type.

    Values:
      RISK_TYPE_UNSPECIFIED: Default unspecified risk. Don't use directly.
      SERVICE_DISRUPTION: Potential service downtime.
      DATA_LOSS: Potential data loss.
      ACCESS_DENY: Potential access denial. The service is still up but some
        or all clients can't access it.
    """
        RISK_TYPE_UNSPECIFIED = 0
        SERVICE_DISRUPTION = 1
        DATA_LOSS = 2
        ACCESS_DENY = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DetailsValue(_messages.Message):
        """Per-recommender projection.

    Messages:
      AdditionalProperty: An additional property for a DetailsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    details = _messages.MessageField('DetailsValue', 1)
    risks = _messages.EnumField('RisksValueListEntryValuesEnum', 2, repeated=True)