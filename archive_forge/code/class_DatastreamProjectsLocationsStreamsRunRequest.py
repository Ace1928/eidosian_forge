from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsRunRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsRunRequest object.

  Fields:
    name: Required. Name of the stream resource to start, in the format:
      projects/{project_id}/locations/{location}/streams/{stream_name}
    runStreamRequest: A RunStreamRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    runStreamRequest = _messages.MessageField('RunStreamRequest', 2)