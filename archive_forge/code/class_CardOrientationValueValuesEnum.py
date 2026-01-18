from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CardOrientationValueValuesEnum(_messages.Enum):
    """Required. Orientation of the card.

    Values:
      CARD_ORIENTATION_UNSPECIFIED: Not specified.
      HORIZONTAL: Horizontal layout.
      VERTICAL: Vertical layout.
    """
    CARD_ORIENTATION_UNSPECIFIED = 0
    HORIZONTAL = 1
    VERTICAL = 2