from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeMoveResponse(_messages.Message):
    """The response message for resource move analysis.

  Fields:
    moveAnalysis: The list of analyses returned from performing the intended
      resource move analysis. The analysis is grouped by different Google
      Cloud services.
  """
    moveAnalysis = _messages.MessageField('MoveAnalysis', 1, repeated=True)