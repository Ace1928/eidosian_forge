from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def _PrepareBoundingPolygonMessage(bounding_polygon):
    """Prepares the bounding polygons message given user's input."""
    bounding_polygon_message = api_utils.GetMessage().BoundingPoly()
    vertices_message = []
    normalized_vertices_message = []
    if 'vertices' in bounding_polygon:
        for vertex in bounding_polygon['vertices']:
            vertex_int = Vertex(vertex['x'], vertex['y'])
            vertices_message.append(api_utils.GetMessage().Vertex(x=vertex_int.x, y=vertex_int.y))
    if 'normalized-vertices' in bounding_polygon:
        for normalized_vertex in bounding_polygon['normalized-vertices']:
            normalized_vertex_float = NormalizedVertex(normalized_vertex['x'], normalized_vertex['y'])
            normalized_vertices_message.append(api_utils.GetMessage().NormalizedVertex(x=normalized_vertex_float.x, y=normalized_vertex_float.y))
    bounding_polygon_message.vertices = vertices_message
    bounding_polygon_message.normalizedVertices = normalized_vertices_message
    return bounding_polygon_message