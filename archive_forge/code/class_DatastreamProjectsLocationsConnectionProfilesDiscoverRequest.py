from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsConnectionProfilesDiscoverRequest(_messages.Message):
    """A DatastreamProjectsLocationsConnectionProfilesDiscoverRequest object.

  Fields:
    discoverConnectionProfileRequest: A DiscoverConnectionProfileRequest
      resource to be passed as the request body.
    parent: Required. The parent resource of the connection profile type. Must
      be in the format `projects/*/locations/*`.
  """
    discoverConnectionProfileRequest = _messages.MessageField('DiscoverConnectionProfileRequest', 1)
    parent = _messages.StringField(2, required=True)