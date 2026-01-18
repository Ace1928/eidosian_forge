from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Threshold(_messages.Message):
    """Defines a threshold for categorizing time series values.

  Enums:
    ColorValueValuesEnum: The state color for this threshold. Color is not
      allowed in a XyChart.
    DirectionValueValuesEnum: The direction for the current threshold.
      Direction is not allowed in a XyChart.
    TargetAxisValueValuesEnum: The target axis to use for plotting the
      threshold. Target axis is not allowed in a Scorecard.

  Fields:
    color: The state color for this threshold. Color is not allowed in a
      XyChart.
    direction: The direction for the current threshold. Direction is not
      allowed in a XyChart.
    label: A label for the threshold.
    targetAxis: The target axis to use for plotting the threshold. Target axis
      is not allowed in a Scorecard.
    value: The value of the threshold. The value should be defined in the
      native scale of the metric.
  """

    class ColorValueValuesEnum(_messages.Enum):
        """The state color for this threshold. Color is not allowed in a XyChart.

    Values:
      COLOR_UNSPECIFIED: Color is unspecified. Not allowed in well-formed
        requests.
      YELLOW: Crossing the threshold is "concerning" behavior.
      RED: Crossing the threshold is "emergency" behavior.
    """
        COLOR_UNSPECIFIED = 0
        YELLOW = 1
        RED = 2

    class DirectionValueValuesEnum(_messages.Enum):
        """The direction for the current threshold. Direction is not allowed in a
    XyChart.

    Values:
      DIRECTION_UNSPECIFIED: Not allowed in well-formed requests.
      ABOVE: The threshold will be considered crossed if the actual value is
        above the threshold value.
      BELOW: The threshold will be considered crossed if the actual value is
        below the threshold value.
    """
        DIRECTION_UNSPECIFIED = 0
        ABOVE = 1
        BELOW = 2

    class TargetAxisValueValuesEnum(_messages.Enum):
        """The target axis to use for plotting the threshold. Target axis is not
    allowed in a Scorecard.

    Values:
      TARGET_AXIS_UNSPECIFIED: The target axis was not specified. Defaults to
        Y1.
      Y1: The y_axis (the right axis of chart).
      Y2: The y2_axis (the left axis of chart).
    """
        TARGET_AXIS_UNSPECIFIED = 0
        Y1 = 1
        Y2 = 2
    color = _messages.EnumField('ColorValueValuesEnum', 1)
    direction = _messages.EnumField('DirectionValueValuesEnum', 2)
    label = _messages.StringField(3)
    targetAxis = _messages.EnumField('TargetAxisValueValuesEnum', 4)
    value = _messages.FloatField(5)