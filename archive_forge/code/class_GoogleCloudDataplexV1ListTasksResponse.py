from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListTasksResponse(_messages.Message):
    """List tasks response.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    tasks: Tasks under the given parent lake.
    unreachableLocations: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    tasks = _messages.MessageField('GoogleCloudDataplexV1Task', 2, repeated=True)
    unreachableLocations = _messages.StringField(3, repeated=True)