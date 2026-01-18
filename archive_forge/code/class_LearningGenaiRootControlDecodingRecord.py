from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootControlDecodingRecord(_messages.Message):
    """Holds one control decoding record.

  Fields:
    prefixes: Prefixes feeded into scorer.
    scores: Per policy scores returned from Scorer. Expect to have the same
      number of scores as in `thresholds`.
    suffiexes: Suffixes feeded into scorer.
    thresholds: Per policy thresholds from user config.
  """
    prefixes = _messages.StringField(1)
    scores = _messages.MessageField('LearningGenaiRootControlDecodingRecordPolicyScore', 2, repeated=True)
    suffiexes = _messages.StringField(3)
    thresholds = _messages.MessageField('LearningGenaiRootControlDecodingConfigThreshold', 4, repeated=True)