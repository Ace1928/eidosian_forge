from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsTransformersCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsTransformersCreateRequest object.

  Fields:
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}`.
    requestId: An optional request ID to identify requests. Specify a unique
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
    transformer: A Transformer resource to be passed as the request body.
    transformerId: Required. The ID of the transformer resource to be created.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    transformer = _messages.MessageField('Transformer', 3)
    transformerId = _messages.StringField(4)