from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringQueryLanguageCondition(_messages.Message):
    """A condition type that allows alert policies to be defined using
  Monitoring Query Language (https://cloud.google.com/monitoring/mql).

  Enums:
    EvaluationMissingDataValueValuesEnum: A condition control that determines
      how metric-threshold conditions are evaluated when data stops arriving.

  Fields:
    duration: The amount of time that a time series must violate the threshold
      to be considered failing. Currently, only values that are a multiple of
      a minute--e.g., 0, 60, 120, or 300 seconds--are supported. If an invalid
      value is given, an error will be returned. When choosing a duration, it
      is useful to keep in mind the frequency of the underlying time series
      data (which may also be affected by any alignments specified in the
      aggregations field); a good duration is long enough so that a single
      outlier does not generate spurious alerts, but short enough that
      unhealthy states are detected and alerted on quickly.
    evaluationMissingData: A condition control that determines how metric-
      threshold conditions are evaluated when data stops arriving.
    query: Monitoring Query Language (https://cloud.google.com/monitoring/mql)
      query that outputs a boolean stream.
    trigger: The number/percent of time series for which the comparison must
      hold in order for the condition to trigger. If unspecified, then the
      condition will trigger if the comparison is true for any of the time
      series that have been identified by filter and aggregations, or by the
      ratio, if denominator_filter and denominator_aggregations are specified.
  """

    class EvaluationMissingDataValueValuesEnum(_messages.Enum):
        """A condition control that determines how metric-threshold conditions
    are evaluated when data stops arriving.

    Values:
      EVALUATION_MISSING_DATA_UNSPECIFIED: An unspecified evaluation missing
        data option. Equivalent to EVALUATION_MISSING_DATA_NO_OP.
      EVALUATION_MISSING_DATA_INACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as false.
      EVALUATION_MISSING_DATA_ACTIVE: If there is no data to evaluate the
        condition, then evaluate the condition as true.
      EVALUATION_MISSING_DATA_NO_OP: Do not evaluate the condition to any
        value if there is no data.
    """
        EVALUATION_MISSING_DATA_UNSPECIFIED = 0
        EVALUATION_MISSING_DATA_INACTIVE = 1
        EVALUATION_MISSING_DATA_ACTIVE = 2
        EVALUATION_MISSING_DATA_NO_OP = 3
    duration = _messages.StringField(1)
    evaluationMissingData = _messages.EnumField('EvaluationMissingDataValueValuesEnum', 2)
    query = _messages.StringField(3)
    trigger = _messages.MessageField('Trigger', 4)