from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBuildsResponse(_messages.Message):
    """Response including listed builds.

  Fields:
    builds: Builds will be sorted by `create_time`, descending.
    nextPageToken: Token to receive the next page of results. This will be
      absent if the end of the response list has been reached.
  """
    builds = _messages.MessageField('Build', 1, repeated=True)
    nextPageToken = _messages.StringField(2)