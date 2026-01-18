from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootSimilarityTakedownPhrase(_messages.Message):
    """Each SimilarityTakedownPhrase treats a logical group of blocked and
  allowed phrases together along with a corresponding punt If the closest
  matching response is of the allowed type, we allow the response If the
  closest matching response is of the blocked type, we block the response. eg:
  Blocked phrase - "All lives matter"

  Fields:
    blockedPhrase: A string attribute.
  """
    blockedPhrase = _messages.StringField(1)