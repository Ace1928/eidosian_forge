from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReferenceImagesResponse(_messages.Message):
    """Response message for the `ListReferenceImages` method.

  Fields:
    nextPageToken: The next_page_token returned from a previous List request,
      if any.
    pageSize: The maximum number of items to return. Default 10, maximum 100.
    referenceImages: The list of reference images.
  """
    nextPageToken = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    referenceImages = _messages.MessageField('ReferenceImage', 3, repeated=True)