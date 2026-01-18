from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeBackendBucketsInsertRequest(_messages.Message):
    """A ComputeBackendBucketsInsertRequest object.

  Fields:
    backendBucket: A BackendBucket resource to be passed as the request body.
    project: Project ID for this request.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      ( 00000000-0000-0000-0000-000000000000).
  """
    backendBucket = _messages.MessageField('BackendBucket', 1)
    project = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)