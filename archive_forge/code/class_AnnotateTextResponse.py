from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotateTextResponse(_messages.Message):
    """The text annotations response message.

  Fields:
    categories: Categories identified in the input document.
    documentSentiment: The overall sentiment for the document. Populated if
      the user enables
      AnnotateTextRequest.Features.extract_document_sentiment.
    entities: Entities, along with their semantic information, in the input
      document. Populated if the user enables
      AnnotateTextRequest.Features.extract_entities.
    language: The language of the text, which will be the same as the language
      specified in the request or, if not specified, the automatically-
      detected language. See Document.language field for more details.
    moderationCategories: Harmful and sensitive categories identified in the
      input document.
    sentences: Sentences in the input document. Populated if the user enables
      AnnotateTextRequest.Features.extract_syntax.
    tokens: Tokens, along with their syntactic information, in the input
      document. Populated if the user enables
      AnnotateTextRequest.Features.extract_syntax.
  """
    categories = _messages.MessageField('ClassificationCategory', 1, repeated=True)
    documentSentiment = _messages.MessageField('Sentiment', 2)
    entities = _messages.MessageField('Entity', 3, repeated=True)
    language = _messages.StringField(4)
    moderationCategories = _messages.MessageField('ClassificationCategory', 5, repeated=True)
    sentences = _messages.MessageField('Sentence', 6, repeated=True)
    tokens = _messages.MessageField('Token', 7, repeated=True)