from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CardWidthValueValuesEnum(_messages.Enum):
    """Required. The width of the cards in the carousel.

    Values:
      CARD_WIDTH_UNSPECIFIED: Not specified.
      SMALL: 120 DP. Note that tall media cannot be used.
      MEDIUM: 232 DP.
    """
    CARD_WIDTH_UNSPECIFIED = 0
    SMALL = 1
    MEDIUM = 2