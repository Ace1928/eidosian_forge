from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotateFileRequest(_messages.Message):
    """A request to annotate one single file, e.g. a PDF, TIFF or GIF file.

  Fields:
    features: Required. Requested features.
    imageContext: Additional context that may accompany the image(s) in the
      file.
    inputConfig: Required. Information about the input file.
    pages: Pages of the file to perform image annotation. Pages starts from 1,
      we assume the first page of the file is page 1. At most 5 pages are
      supported per request. Pages can be negative. Page 1 means the first
      page. Page 2 means the second page. Page -1 means the last page. Page -2
      means the second to the last page. If the file is GIF instead of PDF or
      TIFF, page refers to GIF frames. If this field is empty, by default the
      service performs image annotation for the first 5 pages of the file.
  """
    features = _messages.MessageField('Feature', 1, repeated=True)
    imageContext = _messages.MessageField('ImageContext', 2)
    inputConfig = _messages.MessageField('InputConfig', 3)
    pages = _messages.IntegerField(4, repeated=True, variant=_messages.Variant.INT32)