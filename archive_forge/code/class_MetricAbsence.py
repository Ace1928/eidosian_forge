from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricAbsence(_messages.Message):
    """A condition type that checks that monitored resources are reporting
  data. The configuration defines a metric and a set of monitored resources.
  The predicate is considered in violation when a time series for the
  specified metric of a monitored resource does not include any data in the
  specified duration.

  Fields:
    aggregations: Specifies the alignment of data points in individual time
      series as well as how to combine the retrieved time series together
      (such as when aggregating multiple streams on each resource to a single
      stream for each resource or when aggregating streams across all members
      of a group of resources). Multiple aggregations are applied in the order
      specified.This field is similar to the one in the ListTimeSeries request
      (https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSer
      ies/list). It is advisable to use the ListTimeSeries method when
      debugging this field.
    duration: The amount of time that a time series must fail to report new
      data to be considered failing. The minimum value of this field is 120
      seconds. Larger values that are a multiple of a minute--for example, 240
      or 300 seconds--are supported. If an invalid value is given, an error
      will be returned. The Duration.nanos field is ignored.
    filter: Required. A filter
      (https://cloud.google.com/monitoring/api/v3/filters) that identifies
      which time series should be compared with the threshold.The filter is
      similar to the one that is specified in the ListTimeSeries request (http
      s://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.timeSeries/l
      ist) (that call is useful to verify the time series that will be
      retrieved / processed). The filter must specify the metric type and the
      resource type. Optionally, it can specify resource labels and metric
      labels. This field must not exceed 2048 Unicode characters in length.
    trigger: The number/percent of time series for which the comparison must
      hold in order for the condition to trigger. If unspecified, then the
      condition will trigger if the comparison is true for any of the time
      series that have been identified by filter and aggregations.
  """
    aggregations = _messages.MessageField('Aggregation', 1, repeated=True)
    duration = _messages.StringField(2)
    filter = _messages.StringField(3)
    trigger = _messages.MessageField('Trigger', 4)