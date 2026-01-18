from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOSImagesResponse(_messages.Message):
    """Request for getting all available OS images.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    osImages: The OS images available.
  """
    nextPageToken = _messages.StringField(1)
    osImages = _messages.MessageField('OSImage', 2, repeated=True)