from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceFork(_messages.Message):
    """DEPRECATED in favor of DynamicSourceSplit.

  Fields:
    primary: DEPRECATED
    primarySource: DEPRECATED
    residual: DEPRECATED
    residualSource: DEPRECATED
  """
    primary = _messages.MessageField('SourceSplitShard', 1)
    primarySource = _messages.MessageField('DerivedSource', 2)
    residual = _messages.MessageField('SourceSplitShard', 3)
    residualSource = _messages.MessageField('DerivedSource', 4)