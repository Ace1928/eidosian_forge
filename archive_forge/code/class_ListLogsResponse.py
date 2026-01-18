from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLogsResponse(_messages.Message):
    """Result returned from ListLogs.

  Fields:
    logNames: A list of log names. For example, "projects/my-
      project/logs/syslog" or
      "organizations/123/logs/cloudresourcemanager.googleapis.com%2Factivity".
    nextPageToken: If there might be more results than those appearing in this
      response, then nextPageToken is included. To get the next set of
      results, call this method again using the value of nextPageToken as
      pageToken.
  """
    logNames = _messages.StringField(1, repeated=True)
    nextPageToken = _messages.StringField(2)