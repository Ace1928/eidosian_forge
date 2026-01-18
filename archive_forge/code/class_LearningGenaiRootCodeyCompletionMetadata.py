from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootCodeyCompletionMetadata(_messages.Message):
    """Stores all metadata relating to Completion.

  Fields:
    checkpoints: A LearningGenaiRootCodeyCheckpoint attribute.
  """
    checkpoints = _messages.MessageField('LearningGenaiRootCodeyCheckpoint', 1, repeated=True)