from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeEntitySentimentResponse(_messages.Message):
    """The entity-level sentiment analysis response message.

  Fields:
    entities: The recognized entities in the input document with associated
      sentiments.
    language: The language of the text, which will be the same as the language
      specified in the request or, if not specified, the automatically-
      detected language. See Document.language field for more details.
  """
    entities = _messages.MessageField('Entity', 1, repeated=True)
    language = _messages.StringField(2)