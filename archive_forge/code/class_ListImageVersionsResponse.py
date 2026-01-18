from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListImageVersionsResponse(_messages.Message):
    """The ImageVersions in a project and location.

  Fields:
    imageVersions: The list of supported ImageVersions in a location.
    nextPageToken: The page token used to query for the next page if one
      exists.
  """
    imageVersions = _messages.MessageField('ImageVersion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)