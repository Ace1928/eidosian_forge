from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksApiQueueUpdateRequest(_messages.Message):
    """A CloudtasksApiQueueUpdateRequest object.

  Fields:
    appId: Required. The App ID is supplied as an HTTP parameter. Unlike
      internal usage of App ID, it does not include a region prefix. Rather,
      the App ID represents the Project ID against which to make the request.
    httpBody: A HttpBody resource to be passed as the request body.
  """
    appId = _messages.StringField(1)
    httpBody = _messages.MessageField('HttpBody', 2)