from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActivateSpokeRequest(_messages.Message):
    """The request for HubService.ActivateSpoke.

  Fields:
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server knows to
      ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check to see
      whether the original operation was received. If it was, the server
      ignores the second request. This behavior prevents clients from
      mistakenly creating duplicate commitments. The request ID must be a
      valid UUID, with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    requestId = _messages.StringField(1)