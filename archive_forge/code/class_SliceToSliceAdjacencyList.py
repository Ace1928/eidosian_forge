from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SliceToSliceAdjacencyList(_messages.Message):
    """Adjacency list representation of a slice-to-slice traffic matrix.

  Fields:
    srcTraffic: One entry per slice containing the traffic leaving that slice.
  """
    srcTraffic = _messages.MessageField('SrcSliceTraffic', 1, repeated=True)