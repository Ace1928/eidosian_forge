from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Sentence(_messages.Message):
    """Represents a sentence in the input document.

  Fields:
    sentiment: For calls to AnalyzeSentiment or if
      AnnotateTextRequest.Features.extract_document_sentiment is set to true,
      this field will contain the sentiment for the sentence.
    text: The sentence text.
  """
    sentiment = _messages.MessageField('Sentiment', 1)
    text = _messages.MessageField('TextSpan', 2)