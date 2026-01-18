from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsStreamsObjectsLookupRequest(_messages.Message):
    """A DatastreamProjectsLocationsStreamsObjectsLookupRequest object.

  Fields:
    lookupStreamObjectRequest: A LookupStreamObjectRequest resource to be
      passed as the request body.
    parent: Required. The parent stream that owns the collection of objects.
  """
    lookupStreamObjectRequest = _messages.MessageField('LookupStreamObjectRequest', 1)
    parent = _messages.StringField(2, required=True)