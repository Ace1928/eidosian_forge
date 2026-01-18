from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootCodeyOutput(_messages.Message):
    """Top-level wrapper used to store all things codey-related.

  Fields:
    codeyChatMetadata: A LearningGenaiRootCodeyChatMetadata attribute.
    codeyCompletionMetadata: A LearningGenaiRootCodeyCompletionMetadata
      attribute.
  """
    codeyChatMetadata = _messages.MessageField('LearningGenaiRootCodeyChatMetadata', 1)
    codeyCompletionMetadata = _messages.MessageField('LearningGenaiRootCodeyCompletionMetadata', 2)