from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsObjectsStartBackfillJobRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsObjectsStartBackfillJobRequest
  object.

  Fields:
    object: Required. The name of the stream object resource to start a
      backfill job for.
    startBackfillJobRequest: A StartBackfillJobRequest resource to be passed
      as the request body.
  """
    object = _messages.StringField(1, required=True)
    startBackfillJobRequest = _messages.MessageField('StartBackfillJobRequest', 2)