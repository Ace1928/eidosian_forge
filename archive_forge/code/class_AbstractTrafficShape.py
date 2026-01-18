from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AbstractTrafficShape(_messages.Message):
    """Represents an abstract traffic shape in the traffic matrix. By "traffic
  shape", we mean a list of coordinates and the directed edges of traffic that
  flow between them. By "abstract", we mean that each traffic shape is defined
  relative to 0-indexed coordinates. These abstract coordinates are converted
  to absolute coordinates when instantiated in `traffic_shape_instantiation`.

  Fields:
    allToAllTraffic: All to all traffic shape.
    nToMTraffic: N to m traffic shape
    ringTraffic: Ring traffic shape.
  """
    allToAllTraffic = _messages.MessageField('AllToAllTraffic', 1)
    nToMTraffic = _messages.MessageField('NToMTraffic', 2)
    ringTraffic = _messages.MessageField('RingTraffic', 3)