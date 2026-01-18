from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiDocDocumentationResponse(_messages.Message):
    """The catalog item documentation wrapped with response status, error_code,
  etc.

  Fields:
    data: Output only. The documentation resource.
    errorCode: Output only. Unique error code for the request, if any.
    message: Output only. Description of the operation.
    requestId: Output only. Unique ID of the request.
    status: Output only. Status of the operation.
  """
    data = _messages.MessageField('GoogleCloudApigeeV1ApiDocDocumentation', 1)
    errorCode = _messages.StringField(2)
    message = _messages.StringField(3)
    requestId = _messages.StringField(4)
    status = _messages.StringField(5)