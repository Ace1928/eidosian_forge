from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageTypeValueValuesEnum(_messages.Enum):
    """The image type of this PowerImage.

    Values:
      IMAGE_TYPE_UNSPECIFIED: The type of image is not specified
      STOCK: The image is a stock image.
      IMPORT: The image is an imported image.
      SNAPSHOT: The image is a snapshot image.
      CAPTURE: The image is a captured image.
    """
    IMAGE_TYPE_UNSPECIFIED = 0
    STOCK = 1
    IMPORT = 2
    SNAPSHOT = 3
    CAPTURE = 4