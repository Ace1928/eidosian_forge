from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootGroundingMetadata(_messages.Message):
    """A LearningGenaiRootGroundingMetadata object.

  Fields:
    citations: A LearningGenaiRootGroundingMetadataCitation attribute.
    groundingCancelled: True if grounding is cancelled, for example, no facts
      being retrieved.
    searchQueries: A string attribute.
  """
    citations = _messages.MessageField('LearningGenaiRootGroundingMetadataCitation', 1, repeated=True)
    groundingCancelled = _messages.BooleanField(2)
    searchQueries = _messages.StringField(3, repeated=True)