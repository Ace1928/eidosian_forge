from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationsListResponse(_messages.Message):
    """A response containing a partial list of operations and a page token used
  to build the next request if the request has been truncated.

  Fields:
    nextPageToken: Output only. A token used to continue a truncated list
      request.
    operations: Output only. Operations contained in this list response.
  """
    nextPageToken = _messages.StringField(1)
    operations = _messages.MessageField('Operation', 2, repeated=True)