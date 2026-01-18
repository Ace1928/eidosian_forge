from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListChangedFilesResponse(_messages.Message):
    """Response for ListChangedFiles.

  Fields:
    changedFiles: Note: ChangedFileInfo.from_path is not set here.
      ListChangedFiles does not perform rename/copy detection.  The
      ChangedFileInfo.Type describes the changes from source_context1 to
      source_context2. Thus ADDED would mean a file is not present in
      source_context1 but is present in source_context2.
    nextPageToken: Use as the value of page_token in the next call to obtain
      the next page of results. If empty, there are no more results.
  """
    changedFiles = _messages.MessageField('ChangedFileInfo', 1, repeated=True)
    nextPageToken = _messages.StringField(2)