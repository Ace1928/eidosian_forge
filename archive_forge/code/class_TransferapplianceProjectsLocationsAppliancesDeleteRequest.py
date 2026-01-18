from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsAppliancesDeleteRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsAppliancesDeleteRequest object.

  Fields:
    etag: Strongly validated etag, computed by the server to ensure the client
      has an up-to-date value before proceeding. See
      https://google.aip.dev/154.
    name: Required. Name of the resource.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and t he request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with an expected result, but no actual change is made.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)