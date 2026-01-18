from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiDocResponse(_messages.Message):
    """The catalog item resource wrapped with response status, error_code, etc.

  Fields:
    data: The catalog item resource.
    errorCode: Unique error code for the request, if any.
    message: Description of the operation.
    requestId: Unique ID of the request.
    status: Status of the operation.
  """
    data = _messages.MessageField('GoogleCloudApigeeV1ApiDoc', 1)
    errorCode = _messages.StringField(2)
    message = _messages.StringField(3)
    requestId = _messages.StringField(4)
    status = _messages.StringField(5)