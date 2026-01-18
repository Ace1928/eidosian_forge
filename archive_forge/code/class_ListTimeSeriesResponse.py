from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTimeSeriesResponse(_messages.Message):
    """The ListTimeSeries response.

  Fields:
    executionErrors: Query execution errors that may have caused the time
      series data returned to be incomplete.
    nextPageToken: If there are more results than have been returned, then
      this field is set to a non-empty value. To see the additional results,
      use that value as page_token in the next call to this method.
    timeSeries: One or more time series that match the filter included in the
      request.
    unit: The unit in which all time_series point values are reported. unit
      follows the UCUM format for units as seen in
      https://unitsofmeasure.org/ucum.html. If different time_series have
      different units (for example, because they come from different metric
      types, or a unit is absent), then unit will be "{not_a_unit}".
  """
    executionErrors = _messages.MessageField('Status', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    timeSeries = _messages.MessageField('TimeSeries', 3, repeated=True)
    unit = _messages.StringField(4)