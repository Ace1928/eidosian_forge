from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ThumbnailImageAlignmentValueValuesEnum(_messages.Enum):
    """Required if orientation is horizontal. Image preview alignment for
    standalone cards with horizontal layout.

    Values:
      THUMBNAIL_IMAGE_ALIGNMENT_UNSPECIFIED: Not specified.
      LEFT: Thumbnail preview is left-aligned.
      RIGHT: Thumbnail preview is right-aligned.
    """
    THUMBNAIL_IMAGE_ALIGNMENT_UNSPECIFIED = 0
    LEFT = 1
    RIGHT = 2