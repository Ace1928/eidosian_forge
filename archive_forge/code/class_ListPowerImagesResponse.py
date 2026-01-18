from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPowerImagesResponse(_messages.Message):
    """Response message containing the list of Power images.

  Fields:
    nextPageToken: A token identifying a page of results from the server.
    powerImages: The list of images.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    powerImages = _messages.MessageField('PowerImage', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)