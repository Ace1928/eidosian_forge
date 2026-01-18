from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingQuery(_messages.Message):
    """Describes a Cloud Logging query that can be run in Logs Explorer UI or
  via the logging API.In addition to the query itself, additional information
  may be stored to capture the display configuration and other UI state used
  in association with analysis of query results.

  Fields:
    filter: Required. An advanced query using the Logging Query Language
      (https://cloud.google.com/logging/docs/view/logging-query-language). The
      maximum length of the filter is 20000 characters.
    summaryFieldEnd: Characters will be counted from the end of the string.
    summaryFieldStart: Characters will be counted from the start of the
      string.
    summaryFields: Optional. The set of summary fields to display for this
      saved query.
  """
    filter = _messages.StringField(1)
    summaryFieldEnd = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    summaryFieldStart = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    summaryFields = _messages.MessageField('SummaryField', 4, repeated=True)