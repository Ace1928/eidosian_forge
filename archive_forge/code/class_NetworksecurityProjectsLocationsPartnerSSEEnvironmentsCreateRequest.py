from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsPartnerSSEEnvironmentsCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsPartnerSSEEnvironmentsCreateRequest
  object.

  Fields:
    parent: Required. Value for parent.
    partnerSSEEnvironment: A PartnerSSEEnvironment resource to be passed as
      the request body.
    partnerSseEnvironmentId: Required. Id of the requesting object If auto-
      generating Id server-side, remove this field and
      partner_sse_environment_id from the method_signature of Create RPC
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
  """
    parent = _messages.StringField(1, required=True)
    partnerSSEEnvironment = _messages.MessageField('PartnerSSEEnvironment', 2)
    partnerSseEnvironmentId = _messages.StringField(3)
    requestId = _messages.StringField(4)