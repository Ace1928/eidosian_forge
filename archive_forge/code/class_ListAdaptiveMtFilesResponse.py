from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAdaptiveMtFilesResponse(_messages.Message):
    """The response for listing all AdaptiveMt files under a given dataset.

  Fields:
    adaptiveMtFiles: Output only. The Adaptive MT files.
    nextPageToken: Optional. A token to retrieve a page of results. Pass this
      value in the ListAdaptiveMtFilesRequest.page_token field in the
      subsequent call to `ListAdaptiveMtFiles` method to retrieve the next
      page of results.
  """
    adaptiveMtFiles = _messages.MessageField('AdaptiveMtFile', 1, repeated=True)
    nextPageToken = _messages.StringField(2)