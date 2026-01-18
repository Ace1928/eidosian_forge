from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShapeGeneratedEntry(_messages.Message):
    """Shape-based generator that efficiently encodes uniform traffic shapes
  (e.g. "horizontal ring traffic between all chips in two neighboring
  slices"). A single entry represents all the instantiations of a unique
  traffic shape in the traffic matrix.

  Fields:
    abstractTrafficShape: 0-indexed "abstract" traffic shape. See the
      definition of `AbstractTrafficShape` for details.
    traffic: Anticipated traffic across each edge in the concrete traffic
      shape defined above.
    trafficShapeInstantiation: List of coordinates in which we instantiate
      copies of `abstract_traffic_shape`. Conceptually, each coordinate in
      `traffic_shape_instantiation` represents an offset that converts a
      0-indexed "abstract" traffic shape into a concrete traffic shape with
      absolute coordinates in the traffic matrix.
  """
    abstractTrafficShape = _messages.MessageField('AbstractTrafficShape', 1)
    traffic = _messages.MessageField('Traffic', 2)
    trafficShapeInstantiation = _messages.MessageField('CoordinateList', 3)