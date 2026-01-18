from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PointerLocationValueValuesEnum(_messages.Enum):
    """The pointer location for this widget (also sometimes called a "tail")

    Values:
      POINTER_LOCATION_UNSPECIFIED: No visual pointer
      PL_TOP: Placed in the middle of the top of the widget
      PL_RIGHT: Placed in the middle of the right side of the widget
      PL_BOTTOM: Placed in the middle of the bottom of the widget
      PL_LEFT: Placed in the middle of the left side of the widget
      PL_TOP_LEFT: Placed on the left side of the top of the widget
      PL_TOP_RIGHT: Placed on the right side of the top of the widget
      PL_RIGHT_TOP: Placed on the top of the right side of the widget
      PL_RIGHT_BOTTOM: Placed on the bottom of the right side of the widget
      PL_BOTTOM_RIGHT: Placed on the right side of the bottom of the widget
      PL_BOTTOM_LEFT: Placed on the left side of the bottom of the widget
      PL_LEFT_BOTTOM: Placed on the bottom of the left side of the widget
      PL_LEFT_TOP: Placed on the top of the left side of the widget
    """
    POINTER_LOCATION_UNSPECIFIED = 0
    PL_TOP = 1
    PL_RIGHT = 2
    PL_BOTTOM = 3
    PL_LEFT = 4
    PL_TOP_LEFT = 5
    PL_TOP_RIGHT = 6
    PL_RIGHT_TOP = 7
    PL_RIGHT_BOTTOM = 8
    PL_BOTTOM_RIGHT = 9
    PL_BOTTOM_LEFT = 10
    PL_LEFT_BOTTOM = 11
    PL_LEFT_TOP = 12