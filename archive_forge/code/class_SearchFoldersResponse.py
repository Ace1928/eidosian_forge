from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchFoldersResponse(_messages.Message):
    """The response message for searching folders.

  Fields:
    folders: A possibly paginated folder search results. the specified parent
      resource.
    nextPageToken: A pagination token returned from a previous call to
      `SearchFolders` that indicates from where searching should continue.
      This field is optional.
  """
    folders = _messages.MessageField('Folder', 1, repeated=True)
    nextPageToken = _messages.StringField(2)