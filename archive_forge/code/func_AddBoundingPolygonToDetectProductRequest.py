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
def AddBoundingPolygonToDetectProductRequest(ref, args, request):
    """Adds the boundingPoly field to detect product request."""
    del ref
    try:
        single_request = request.requests[0]
    except IndexError:
        return request
    if not args.IsSpecified('bounding_polygon'):
        return request
    polygon = _ValidateAndExtractFromBoundingPolygonArgs(args.bounding_polygon)
    if not polygon:
        return request
    single_request = _InstantiateProductSearchParams(single_request)
    product_search_params = single_request.imageContext.productSearchParams
    if not product_search_params.boundingPoly:
        product_search_params.boundingPoly = api_utils.GetMessage().BoundingPoly()
    bounding_poly = product_search_params.boundingPoly
    if isinstance(polygon[0], Vertex):
        vertices = [api_utils.GetMessage().Vertex(x=v.x, y=v.y) for v in polygon]
        bounding_poly.vertices = vertices
    else:
        normalized_vertices = [api_utils.GetMessage().NormalizedVertex(x=v.x, y=v.y) for v in polygon]
        bounding_poly.normalizedVertices = normalized_vertices
    return request