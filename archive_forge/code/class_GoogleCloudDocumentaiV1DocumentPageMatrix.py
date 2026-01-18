from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageMatrix(_messages.Message):
    """Representation for transformation matrix, intended to be compatible and
  used with OpenCV format for image manipulation.

  Fields:
    cols: Number of columns in the matrix.
    data: The matrix data.
    rows: Number of rows in the matrix.
    type: This encodes information about what data type the matrix uses. For
      example, 0 (CV_8U) is an unsigned 8-bit image. For the full list of
      OpenCV primitive data types, please refer to
      https://docs.opencv.org/4.3.0/d1/d1b/group__core__hal__interface.html
  """
    cols = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    data = _messages.BytesField(2)
    rows = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    type = _messages.IntegerField(4, variant=_messages.Variant.INT32)