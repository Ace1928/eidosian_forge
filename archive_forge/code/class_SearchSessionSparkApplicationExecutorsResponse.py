from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationExecutorsResponse(_messages.Message):
    """List of Executors associated with a Spark Application.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent
      SearchSessionSparkApplicationExecutorsRequest.
    sparkApplicationExecutors: Details about executors used by the
      application.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationExecutors = _messages.MessageField('ExecutorSummary', 2, repeated=True)