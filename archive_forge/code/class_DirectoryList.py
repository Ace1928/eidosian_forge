from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DirectoryList(_messages.Message):
    """A DirectoryList object.

  Messages:
    ItemsValueListEntry: A ItemsValueListEntry object.

  Fields:
    discoveryVersion: Indicate the version of the Discovery API used to
      generate this doc.
    items: The individual directory entries. One entry per api/version pair.
    kind: The kind for this response.
  """

    class ItemsValueListEntry(_messages.Message):
        """A ItemsValueListEntry object.

    Messages:
      IconsValue: Links to 16x16 and 32x32 icons representing the API.

    Fields:
      description: The description of this API.
      discoveryLink: A link to the discovery document.
      discoveryRestUrl: The url for the discovery REST document.
      documentationLink: A link to human readable documentation for the API.
      icons: Links to 16x16 and 32x32 icons representing the API.
      id: The id of this API.
      kind: The kind for this response.
      labels: Labels for the status of this API, such as labs or deprecated.
      name: The name of the API.
      preferred: True if this version is the preferred version to use.
      title: The title of this API.
      version: The version of the API.
    """

        class IconsValue(_messages.Message):
            """Links to 16x16 and 32x32 icons representing the API.

      Fields:
        x16: The url of the 16x16 icon.
        x32: The url of the 32x32 icon.
      """
            x16 = _messages.StringField(1)
            x32 = _messages.StringField(2)
        description = _messages.StringField(1)
        discoveryLink = _messages.StringField(2)
        discoveryRestUrl = _messages.StringField(3)
        documentationLink = _messages.StringField(4)
        icons = _messages.MessageField('IconsValue', 5)
        id = _messages.StringField(6)
        kind = _messages.StringField(7, default='discovery#directoryItem')
        labels = _messages.StringField(8, repeated=True)
        name = _messages.StringField(9)
        preferred = _messages.BooleanField(10)
        title = _messages.StringField(11)
        version = _messages.StringField(12)
    discoveryVersion = _messages.StringField(1, default='v1')
    items = _messages.MessageField('ItemsValueListEntry', 2, repeated=True)
    kind = _messages.StringField(3, default='discovery#directoryList')