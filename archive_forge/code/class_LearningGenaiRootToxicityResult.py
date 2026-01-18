from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootToxicityResult(_messages.Message):
    """A model can generate multiple signals and this captures all the
  generated signals for a single message.

  Fields:
    signals: A LearningGenaiRootToxicitySignal attribute.
  """
    signals = _messages.MessageField('LearningGenaiRootToxicitySignal', 1, repeated=True)