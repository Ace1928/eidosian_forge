from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListChangedFilesRequest(_messages.Message):
    """Request for ListChangedFiles.

  Fields:
    pageSize: The maximum number of ChangedFileInfo values to return.
    pageToken: The value of next_page_token from the previous call. Omit for
      the first page.
    sourceContext1: The starting source context to compare.
    sourceContext2: The ending source context to compare.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    sourceContext1 = _messages.MessageField('SourceContext', 3)
    sourceContext2 = _messages.MessageField('SourceContext', 4)