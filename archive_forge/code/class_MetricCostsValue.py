from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MetricCostsValue(_messages.Message):
    """Metrics to update when the selected methods are called, and the
    associated cost applied to each metric.  The key of the map is the metric
    name, and the values are the amount increased for the metric against which
    the quota limits are defined. The value must not be negative.

    Messages:
      AdditionalProperty: An additional property for a MetricCostsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MetricCostsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MetricCostsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)