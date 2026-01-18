from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModerateTextResponse(_messages.Message):
    """The document moderation response message.

  Fields:
    moderationCategories: Harmful and sensitive categories representing the
      input document.
  """
    moderationCategories = _messages.MessageField('ClassificationCategory', 1, repeated=True)