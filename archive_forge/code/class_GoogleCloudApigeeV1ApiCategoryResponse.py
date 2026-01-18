from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiCategoryResponse(_messages.Message):
    """The API category resource wrapped with response status, error_code, etc.

  Fields:
    data: The API category resource.
    errorCode: Unique error code for the request, if any.
    message: Description of the operation.
    requestId: Unique ID of the request.
    status: Status of the operation.
  """
    data = _messages.MessageField('GoogleCloudApigeeV1ApiCategory', 1)
    errorCode = _messages.StringField(2)
    message = _messages.StringField(3)
    requestId = _messages.StringField(4)
    status = _messages.StringField(5)