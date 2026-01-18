from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesDeleteRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesDeleteRequest object.

  Fields:
    allowMissing: Optional. If true and the Service is not found, the request
      will succeed but no action will be taken on the server.
    etag: Optional. The etag of the Service. If this is provided, it must
      match the server's etag.
    force: Optional. If set to true, any child resources (i.e. Schema,
      SchemaRevisions, Connectors, and ConnectorRevisions) will also be
      deleted. Otherwise, the request will only work if the Service has no
      child resources.
    name: Required. The name of the service to delete, in the format: ```
      projects/{project}/locations/{location}/services/{service} ```
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
    validateOnly: Optional. If set, validate the request and preview the
      Service, but do not actually delete it.
  """
    allowMissing = _messages.BooleanField(1)
    etag = _messages.StringField(2)
    force = _messages.BooleanField(3)
    name = _messages.StringField(4, required=True)
    requestId = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)