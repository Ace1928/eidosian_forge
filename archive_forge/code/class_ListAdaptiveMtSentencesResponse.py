from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAdaptiveMtSentencesResponse(_messages.Message):
    """List AdaptiveMt sentences response.

  Fields:
    adaptiveMtSentences: Output only. The list of AdaptiveMtSentences.
    nextPageToken: Optional.
  """
    adaptiveMtSentences = _messages.MessageField('AdaptiveMtSentence', 1, repeated=True)
    nextPageToken = _messages.StringField(2)