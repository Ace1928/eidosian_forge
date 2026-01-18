from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrientationValueValuesEnum(_messages.Enum):
    """Detected orientation for the Layout.

    Values:
      ORIENTATION_UNSPECIFIED: Unspecified orientation.
      PAGE_UP: Orientation is aligned with page up.
      PAGE_RIGHT: Orientation is aligned with page right. Turn the head 90
        degrees clockwise from upright to read.
      PAGE_DOWN: Orientation is aligned with page down. Turn the head 180
        degrees from upright to read.
      PAGE_LEFT: Orientation is aligned with page left. Turn the head 90
        degrees counterclockwise from upright to read.
    """
    ORIENTATION_UNSPECIFIED = 0
    PAGE_UP = 1
    PAGE_RIGHT = 2
    PAGE_DOWN = 3
    PAGE_LEFT = 4