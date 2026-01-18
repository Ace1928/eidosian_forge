from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1ImageAnnotationContext(_messages.Message):
    """If an image was produced from a file (e.g. a PDF), this message gives
  information about the source of that image.

  Fields:
    pageNumber: If the file was a PDF or TIFF, this field gives the page
      number within the file used to produce the image.
    uri: The URI of the file used to produce the image.
  """
    pageNumber = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    uri = _messages.StringField(2)