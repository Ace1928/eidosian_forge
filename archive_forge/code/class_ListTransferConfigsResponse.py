from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTransferConfigsResponse(_messages.Message):
    """The returned list of pipelines in the project.

  Fields:
    nextPageToken: Output only. The next-pagination token. For multiple-page
      list results, this token can be used as the
      `ListTransferConfigsRequest.page_token` to request the next page of list
      results.
    transferConfigs: Output only. The stored pipeline transfer configurations.
  """
    nextPageToken = _messages.StringField(1)
    transferConfigs = _messages.MessageField('TransferConfig', 2, repeated=True)