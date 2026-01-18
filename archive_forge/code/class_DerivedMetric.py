from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DerivedMetric(_messages.Message):
    """A message representing a derived metric.

  Fields:
    denominator: The name of the denominator metric. e.g. "rows".
    numerator: The name of the numerator metric. e.g. "latency".
  """
    denominator = _messages.MessageField('LocalizedString', 1)
    numerator = _messages.MessageField('LocalizedString', 2)