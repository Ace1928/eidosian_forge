from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReadResponse(_messages.Message):
    """Response to read request. Exactly one of entries, file or
  external_reference will be populated, depending on what the path in the
  request denotes.

  Fields:
    entries: Contains the directory entries if the request specifies a
      directory.
    externalReference: The read path denotes a Git submodule.
    file: Contains file metadata and contents if the request specifies a file.
    nextPageToken: Use as the value of page_token in the next call to obtain
      the next page of results. If empty, there are no more results.
    sourceContext: Returns the SourceContext actually used, resolving any
      alias in the input SourceContext into its revision ID and returning the
      actual current snapshot ID if the read was from a workspace with an
      unspecified snapshot ID.
  """
    entries = _messages.MessageField('DirectoryEntry', 1, repeated=True)
    externalReference = _messages.MessageField('ExternalReference', 2)
    file = _messages.MessageField('File', 3)
    nextPageToken = _messages.StringField(4)
    sourceContext = _messages.MessageField('SourceContext', 5)