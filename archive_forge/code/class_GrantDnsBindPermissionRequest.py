from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrantDnsBindPermissionRequest(_messages.Message):
    """Request message for VmwareEngine.GrantDnsBindPermission

  Fields:
    etag: Optional. Checksum used to ensure that the user-provided value is up
      to date before the server processes the request. The server compares
      provided checksum with the current checksum of the resource. If the
      user-provided value is out of date, this request returns an `ABORTED`
      error.
    principal: Required. The consumer provided user/service account which
      needs to be granted permission to bind with the intranet VPC
      corresponding to the consumer project.
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check if
      original operation with the same request ID was received, and if so,
      will ignore the second request. This prevents clients from accidentally
      creating duplicate commitments. The request ID must be a valid UUID with
      the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    etag = _messages.StringField(1)
    principal = _messages.MessageField('Principal', 2)
    requestId = _messages.StringField(3)