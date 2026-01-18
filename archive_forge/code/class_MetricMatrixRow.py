from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricMatrixRow(_messages.Message):
    """A message representing a row of a matrix of floats.

  Fields:
    cols: The columns of the row.
  """
    cols = _messages.FloatField(1, repeated=True, variant=_messages.Variant.FLOAT)