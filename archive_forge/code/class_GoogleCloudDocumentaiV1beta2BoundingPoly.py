from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2BoundingPoly(_messages.Message):
    """A bounding polygon for the detected image annotation.

  Fields:
    normalizedVertices: The bounding polygon normalized vertices.
    vertices: The bounding polygon vertices.
  """
    normalizedVertices = _messages.MessageField('GoogleCloudDocumentaiV1beta2NormalizedVertex', 1, repeated=True)
    vertices = _messages.MessageField('GoogleCloudDocumentaiV1beta2Vertex', 2, repeated=True)