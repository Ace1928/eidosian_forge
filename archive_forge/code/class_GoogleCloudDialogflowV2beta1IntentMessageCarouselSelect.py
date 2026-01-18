from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageCarouselSelect(_messages.Message):
    """The card for presenting a carousel of options to select from.

  Fields:
    items: Required. Carousel items.
  """
    items = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessageCarouselSelectItem', 1, repeated=True)