from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationStageAttemptTasksResponse(_messages.Message):
    """List of tasks for a stage of a Spark Application

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent
      SearchSessionSparkApplicationStageAttemptTasksRequest.
    sparkApplicationStageAttemptTasks: Output only. Data corresponding to
      tasks created by spark.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationStageAttemptTasks = _messages.MessageField('TaskData', 2, repeated=True)