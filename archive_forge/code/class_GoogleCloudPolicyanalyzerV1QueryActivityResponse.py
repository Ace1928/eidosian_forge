from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicyanalyzerV1QueryActivityResponse(_messages.Message):
    """Response to the `QueryActivity` method.

  Fields:
    activities: The set of activities that match the filter included in the
      request.
    nextPageToken: If there might be more results than those appearing in this
      response, then `nextPageToken` is included. To get the next set of
      results, call this method again using the value of `nextPageToken` as
      `pageToken`.
  """
    activities = _messages.MessageField('GoogleCloudPolicyanalyzerV1Activity', 1, repeated=True)
    nextPageToken = _messages.StringField(2)