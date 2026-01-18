from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartOptions(_messages.Message):
    """Options to control visual rendering of a chart.

  Enums:
    ModeValueValuesEnum: The chart mode.

  Fields:
    displayHorizontal: Preview: Configures whether the charted values are
      shown on the horizontal or vertical axis. By default, values are
      represented the vertical axis. This is a preview feature and may be
      subject to change before final release.
    mode: The chart mode.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """The chart mode.

    Values:
      MODE_UNSPECIFIED: Mode is unspecified. The view will default to COLOR.
      COLOR: The chart distinguishes data series using different color. Line
        colors may get reused when there are many lines in the chart.
      X_RAY: The chart uses the Stackdriver x-ray mode, in which each data set
        is plotted using the same semi-transparent color.
      STATS: The chart displays statistics such as average, median, 95th
        percentile, and more.
    """
        MODE_UNSPECIFIED = 0
        COLOR = 1
        X_RAY = 2
        STATS = 3
    displayHorizontal = _messages.BooleanField(1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)