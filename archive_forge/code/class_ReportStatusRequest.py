from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportStatusRequest(_messages.Message):
    """Request report the connector status.

  Fields:
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    resourceInfo: Required. Resource info of the connector.
    validateOnly: Optional. If set, validates request by executing a dry-run
      which would not alter the resource in any way.
  """
    requestId = _messages.StringField(1)
    resourceInfo = _messages.MessageField('ResourceInfo', 2)
    validateOnly = _messages.BooleanField(3)