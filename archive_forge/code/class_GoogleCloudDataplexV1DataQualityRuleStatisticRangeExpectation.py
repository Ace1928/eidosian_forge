from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityRuleStatisticRangeExpectation(_messages.Message):
    """Evaluates whether the column aggregate statistic lies between a
  specified range.

  Enums:
    StatisticValueValuesEnum: Optional. The aggregate metric to evaluate.

  Fields:
    maxValue: Optional. The maximum column statistic value allowed for a row
      to pass this validation.At least one of min_value and max_value need to
      be provided.
    minValue: Optional. The minimum column statistic value allowed for a row
      to pass this validation.At least one of min_value and max_value need to
      be provided.
    statistic: Optional. The aggregate metric to evaluate.
    strictMaxEnabled: Optional. Whether column statistic needs to be strictly
      lesser than ('<') the maximum, or if equality is allowed.Only relevant
      if a max_value has been defined. Default = false.
    strictMinEnabled: Optional. Whether column statistic needs to be strictly
      greater than ('>') the minimum, or if equality is allowed.Only relevant
      if a min_value has been defined. Default = false.
  """

    class StatisticValueValuesEnum(_messages.Enum):
        """Optional. The aggregate metric to evaluate.

    Values:
      STATISTIC_UNDEFINED: Unspecified statistic type
      MEAN: Evaluate the column mean
      MIN: Evaluate the column min
      MAX: Evaluate the column max
    """
        STATISTIC_UNDEFINED = 0
        MEAN = 1
        MIN = 2
        MAX = 3
    maxValue = _messages.StringField(1)
    minValue = _messages.StringField(2)
    statistic = _messages.EnumField('StatisticValueValuesEnum', 3)
    strictMaxEnabled = _messages.BooleanField(4)
    strictMinEnabled = _messages.BooleanField(5)