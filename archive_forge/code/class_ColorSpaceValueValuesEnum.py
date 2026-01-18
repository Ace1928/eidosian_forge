from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColorSpaceValueValuesEnum(_messages.Enum):
    """Enums for color space, used for processing images in Object Table. See
    more details at https://www.tensorflow.org/io/tutorials/colorspace.

    Values:
      COLOR_SPACE_UNSPECIFIED: Unspecified color space
      RGB: RGB
      HSV: HSV
      YIQ: YIQ
      YUV: YUV
      GRAYSCALE: GRAYSCALE
    """
    COLOR_SPACE_UNSPECIFIED = 0
    RGB = 1
    HSV = 2
    YIQ = 3
    YUV = 4
    GRAYSCALE = 5