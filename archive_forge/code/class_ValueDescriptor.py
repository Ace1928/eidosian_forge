from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValueDescriptor(_messages.Message):
    """A descriptor for the value columns in a data point.

  Enums:
    MetricKindValueValuesEnum: The value stream kind.
    ValueTypeValueValuesEnum: The value type.

  Fields:
    key: The value key.
    metricKind: The value stream kind.
    unit: The unit in which time_series point values are reported. unit
      follows the UCUM format for units as seen in
      https://unitsofmeasure.org/ucum.html. unit is only valid if value_type
      is INTEGER, DOUBLE, DISTRIBUTION.
    valueType: The value type.
  """

    class MetricKindValueValuesEnum(_messages.Enum):
        """The value stream kind.

    Values:
      METRIC_KIND_UNSPECIFIED: Do not use this default value.
      GAUGE: An instantaneous measurement of a value.
      DELTA: The change in a value during a time interval.
      CUMULATIVE: A value accumulated over a time interval. Cumulative
        measurements in a time series should have the same start time and
        increasing end times, until an event resets the cumulative value to
        zero and sets a new start time for the following points.
    """
        METRIC_KIND_UNSPECIFIED = 0
        GAUGE = 1
        DELTA = 2
        CUMULATIVE = 3

    class ValueTypeValueValuesEnum(_messages.Enum):
        """The value type.

    Values:
      VALUE_TYPE_UNSPECIFIED: Do not use this default value.
      BOOL: The value is a boolean. This value type can be used only if the
        metric kind is GAUGE.
      INT64: The value is a signed 64-bit integer.
      DOUBLE: The value is a double precision floating point number.
      STRING: The value is a text string. This value type can be used only if
        the metric kind is GAUGE.
      DISTRIBUTION: The value is a Distribution.
      MONEY: The value is money.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        BOOL = 1
        INT64 = 2
        DOUBLE = 3
        STRING = 4
        DISTRIBUTION = 5
        MONEY = 6
    key = _messages.StringField(1)
    metricKind = _messages.EnumField('MetricKindValueValuesEnum', 2)
    unit = _messages.StringField(3)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 4)