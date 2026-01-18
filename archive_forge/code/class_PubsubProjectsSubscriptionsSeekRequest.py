from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsSeekRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsSeekRequest object.

  Fields:
    seekRequest: A SeekRequest resource to be passed as the request body.
    subscription: Required. The subscription to affect.
  """
    seekRequest = _messages.MessageField('SeekRequest', 1)
    subscription = _messages.StringField(2, required=True)