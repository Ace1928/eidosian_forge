from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricUpdate(_messages.Message):
    """Describes the state of a metric.

  Fields:
    cumulative: True if this metric is reported as the total cumulative
      aggregate value accumulated since the worker started working on this
      WorkItem. By default this is false, indicating that this metric is
      reported as a delta that is not associated with any WorkItem.
    distribution: A struct value describing properties of a distribution of
      numeric values.
    gauge: A struct value describing properties of a Gauge. Metrics of gauge
      type show the value of a metric across time, and is aggregated based on
      the newest value.
    internal: Worker-computed aggregate value for internal use by the Dataflow
      service.
    kind: Metric aggregation kind. The possible metric aggregation kinds are
      "Sum", "Max", "Min", "Mean", "Set", "And", "Or", and "Distribution". The
      specified aggregation kind is case-insensitive. If omitted, this is not
      an aggregated value but instead a single metric sample value.
    meanCount: Worker-computed aggregate value for the "Mean" aggregation
      kind. This holds the count of the aggregated values and is used in
      combination with mean_sum above to obtain the actual mean aggregate
      value. The only possible value type is Long.
    meanSum: Worker-computed aggregate value for the "Mean" aggregation kind.
      This holds the sum of the aggregated values and is used in combination
      with mean_count below to obtain the actual mean aggregate value. The
      only possible value types are Long and Double.
    name: Name of the metric.
    scalar: Worker-computed aggregate value for aggregation kinds "Sum",
      "Max", "Min", "And", and "Or". The possible value types are Long,
      Double, and Boolean.
    set: Worker-computed aggregate value for the "Set" aggregation kind. The
      only possible value type is a list of Values whose type can be Long,
      Double, or String, according to the metric's type. All Values in the
      list must be of the same type.
    updateTime: Timestamp associated with the metric value. Optional when
      workers are reporting work progress; it will be filled in responses from
      the metrics API.
  """
    cumulative = _messages.BooleanField(1)
    distribution = _messages.MessageField('extra_types.JsonValue', 2)
    gauge = _messages.MessageField('extra_types.JsonValue', 3)
    internal = _messages.MessageField('extra_types.JsonValue', 4)
    kind = _messages.StringField(5)
    meanCount = _messages.MessageField('extra_types.JsonValue', 6)
    meanSum = _messages.MessageField('extra_types.JsonValue', 7)
    name = _messages.MessageField('MetricStructuredName', 8)
    scalar = _messages.MessageField('extra_types.JsonValue', 9)
    set = _messages.MessageField('extra_types.JsonValue', 10)
    updateTime = _messages.StringField(11)