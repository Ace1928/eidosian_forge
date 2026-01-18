from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageFoldersListRequest(_messages.Message):
    """A StorageFoldersListRequest object.

  Fields:
    bucket: Name of the bucket in which to look for folders.
    delimiter: Returns results in a directory-like mode. The only supported
      value is '/'. If set, items will only contain folders that either
      exactly match the prefix, or are one level below the prefix.
    endOffset: Filter results to folders whose names are lexicographically
      before endOffset. If startOffset is also set, the folders listed will
      have names between startOffset (inclusive) and endOffset (exclusive).
    pageSize: Maximum number of items to return in a single page of responses.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.
    prefix: Filter results to folders whose paths begin with this prefix. If
      set, the value must either be an empty string or end with a '/'.
    startOffset: Filter results to folders whose names are lexicographically
      equal to or after startOffset. If endOffset is also set, the folders
      listed will have names between startOffset (inclusive) and endOffset
      (exclusive).
  """
    bucket = _messages.StringField(1, required=True)
    delimiter = _messages.StringField(2)
    endOffset = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    prefix = _messages.StringField(6)
    startOffset = _messages.StringField(7)