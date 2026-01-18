from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocgdcProjectsLocationsServiceInstancesDeleteRequest(_messages.Message):
    """A DataprocgdcProjectsLocationsServiceInstancesDeleteRequest object.

  Fields:
    allowMissing: Optional. If set to true, and the service instance is not
      found, the request will succeed but no action will be taken on the
      server
    etag: Optional. The etag of the service instance. If this is provided, it
      must match the server etag.
    force: Optional. If set to true, any jobs and job environments associated
      with this service instance will also be deleted. If false (default) the
      service instance can only be deleted if there are no job environments or
      jobs associated with the service instance.
    name: Required. Name of the resource
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)