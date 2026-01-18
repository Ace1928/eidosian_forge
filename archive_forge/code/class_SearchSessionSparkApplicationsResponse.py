from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationsResponse(_messages.Message):
    """A list of summary of Spark Applications

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent SearchSessionSparkApplicationsRequest.
    sparkApplications: Output only. High level information corresponding to an
      application.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplications = _messages.MessageField('SparkApplication', 2, repeated=True)