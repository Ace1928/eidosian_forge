from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListKmsConfigsResponse(_messages.Message):
    """ListKmsConfigsResponse is the response to a ListKmsConfigsRequest.

  Fields:
    kmsConfigs: The list of KmsConfigs
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    kmsConfigs = _messages.MessageField('KmsConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)