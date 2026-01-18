from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateCollectdTimeSeriesResponse(_messages.Message):
    """The CreateCollectdTimeSeries response.

  Fields:
    payloadErrors: Records the error status for points that were not written
      due to an error in the request.Failed requests for which nothing is
      written will return an error response instead. Requests where data
      points were rejected by the backend will set summary instead.
    summary: Aggregate statistics from writing the payloads. This field is
      omitted if all points were successfully written, so that the response is
      empty. This is for backwards compatibility with clients that log errors
      on any non-empty response.
  """
    payloadErrors = _messages.MessageField('CollectdPayloadError', 1, repeated=True)
    summary = _messages.MessageField('CreateTimeSeriesSummary', 2)