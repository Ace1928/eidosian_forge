from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1MetricValue(_messages.Message):
    """Represents a single metric value.

  Messages:
    LabelsValue: The labels describing the metric value. See comments on
      google.api.servicecontrol.v1.Operation.labels for the overriding
      relationship. Note that this map must not contain monitored resource
      labels.

  Fields:
    boolValue: A boolean value.
    distributionValue: A distribution value.
    doubleValue: A double precision floating point value.
    endTime: The end of the time period over which this metric value's
      measurement applies. If not specified,
      google.api.servicecontrol.v1.Operation.end_time will be used.
    int64Value: A signed 64-bit integer value.
    labels: The labels describing the metric value. See comments on
      google.api.servicecontrol.v1.Operation.labels for the overriding
      relationship. Note that this map must not contain monitored resource
      labels.
    moneyValue: A money value.
    startTime: The start of the time period over which this metric value's
      measurement applies. The time period has different semantics for
      different metric types (cumulative, delta, and gauge). See the metric
      definition documentation in the service configuration for details. If
      not specified, google.api.servicecontrol.v1.Operation.start_time will be
      used.
    stringValue: A text string value.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels describing the metric value. See comments on
    google.api.servicecontrol.v1.Operation.labels for the overriding
    relationship. Note that this map must not contain monitored resource
    labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    boolValue = _messages.BooleanField(1)
    distributionValue = _messages.MessageField('GoogleApiServicecontrolV1Distribution', 2)
    doubleValue = _messages.FloatField(3)
    endTime = _messages.StringField(4)
    int64Value = _messages.IntegerField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    moneyValue = _messages.MessageField('Money', 7)
    startTime = _messages.StringField(8)
    stringValue = _messages.StringField(9)