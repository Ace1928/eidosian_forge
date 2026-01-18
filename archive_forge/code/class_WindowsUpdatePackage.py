from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsUpdatePackage(_messages.Message):
    """Details related to a Windows Update package. Field data and names are
  taken from Windows Update API IUpdate Interface:
  https://docs.microsoft.com/en-us/windows/win32/api/_wua/ Descriptive fields
  like title, and description are localized based on the locale of the VM
  being updated.

  Fields:
    categories: The categories that are associated with this update package.
    description: The localized description of the update package.
    kbArticleIds: A collection of Microsoft Knowledge Base article IDs that
      are associated with the update package.
    lastDeploymentChangeTime: The last published date of the update, in (UTC)
      date and time.
    moreInfoUrls: A collection of URLs that provide more information about the
      update package.
    revisionNumber: The revision number of this update package.
    supportUrl: A hyperlink to the language-specific support information for
      the update.
    title: The localized title of the update package.
    updateId: Gets the identifier of an update package. Stays the same across
      revisions.
  """
    categories = _messages.MessageField('WindowsUpdateCategory', 1, repeated=True)
    description = _messages.StringField(2)
    kbArticleIds = _messages.StringField(3, repeated=True)
    lastDeploymentChangeTime = _messages.StringField(4)
    moreInfoUrls = _messages.StringField(5, repeated=True)
    revisionNumber = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    supportUrl = _messages.StringField(7)
    title = _messages.StringField(8)
    updateId = _messages.StringField(9)