from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1NormalizedBoundingPoly(_messages.Message):
    """Normalized bounding polygon for text (that might not be aligned with
  axis). Contains list of the corner points in clockwise order starting from
  top-left corner. For example, for a rectangular bounding box: When the text
  is horizontal it might look like: 0----1 | | 3----2 When it's clockwise
  rotated 180 degrees around the top-left corner it becomes: 2----3 | | 1----0
  and the vertex order will still be (0, 1, 2, 3). Note that values can be
  less than 0, or greater than 1 due to trignometric calculations for location
  of the box.

  Fields:
    vertices: Normalized vertices of the bounding polygon.
  """
    vertices = _messages.MessageField('GoogleCloudVideointelligenceV1NormalizedVertex', 1, repeated=True)