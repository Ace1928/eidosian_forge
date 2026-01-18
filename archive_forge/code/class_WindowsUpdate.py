from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsUpdate(_messages.Message):
    """Windows Update represents the metadata about the update for the Windows
  operating system. The fields in this message come from the Windows Update
  API documented at https://docs.microsoft.com/en-
  us/windows/win32/api/wuapi/nn-wuapi-iupdate.

  Fields:
    categories: The list of categories to which the update belongs.
    description: The localized description of the update.
    identity: Required - The unique identifier for the update.
    kbArticleIds: The Microsoft Knowledge Base article IDs that are associated
      with the update.
    lastPublishedTimestamp: The last published timestamp of the update.
    supportUrl: The hyperlink to the support information for the update.
    title: The localized title of the update.
  """
    categories = _messages.MessageField('Category', 1, repeated=True)
    description = _messages.StringField(2)
    identity = _messages.MessageField('Identity', 3)
    kbArticleIds = _messages.StringField(4, repeated=True)
    lastPublishedTimestamp = _messages.StringField(5)
    supportUrl = _messages.StringField(6)
    title = _messages.StringField(7)