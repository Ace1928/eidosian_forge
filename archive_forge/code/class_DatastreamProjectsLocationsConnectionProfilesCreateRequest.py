from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsConnectionProfilesCreateRequest(_messages.Message):
    """A DatastreamProjectsLocationsConnectionProfilesCreateRequest object.

  Fields:
    connectionProfile: A ConnectionProfile resource to be passed as the
      request body.
    connectionProfileId: Required. The connection profile identifier.
    force: Optional. Create the connection profile without validating it.
    parent: Required. The parent that owns the collection of
      ConnectionProfiles.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server will
      guarantee that for at least 60 minutes since the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. Only validate the connection profile, but don't
      create any resources. The default is false.
  """
    connectionProfile = _messages.MessageField('ConnectionProfile', 1)
    connectionProfileId = _messages.StringField(2)
    force = _messages.BooleanField(3)
    parent = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)