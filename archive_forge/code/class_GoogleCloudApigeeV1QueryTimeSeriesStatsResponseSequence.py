from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryTimeSeriesStatsResponseSequence(_messages.Message):
    """A sequence of time series.

  Messages:
    DimensionsValue: Map of dimensions and their values that uniquely
      identifies a time series sequence.
    PointsValueListEntry: Single entry in a PointsValue.

  Fields:
    dimensions: Map of dimensions and their values that uniquely identifies a
      time series sequence.
    points: List of points. First value of each inner list is a timestamp.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """Map of dimensions and their values that uniquely identifies a time
    series sequence.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    class PointsValueListEntry(_messages.Message):
        """Single entry in a PointsValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    dimensions = _messages.MessageField('DimensionsValue', 1)
    points = _messages.MessageField('PointsValueListEntry', 2, repeated=True)