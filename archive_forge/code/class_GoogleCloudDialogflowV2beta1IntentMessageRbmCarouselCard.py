from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageRbmCarouselCard(_messages.Message):
    """Carousel Rich Business Messaging (RBM) rich card. Rich cards allow you
  to respond to users with more vivid content, e.g. with media and
  suggestions. If you want to show a single card with more control over the
  layout, please use RbmStandaloneCard instead.

  Enums:
    CardWidthValueValuesEnum: Required. The width of the cards in the
      carousel.

  Fields:
    cardContents: Required. The cards in the carousel. A carousel must have at
      least 2 cards and at most 10.
    cardWidth: Required. The width of the cards in the carousel.
  """

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
    cardContents = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageRbmCardContent', 1, repeated=True)
    cardWidth = _messages.EnumField('CardWidthValueValuesEnum', 2)