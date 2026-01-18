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
def AddBoundingPolygonsToReferenceImageCreationRequest(ref, args, request):
    """Populate the boundingPolygon message."""
    del ref
    if not args.IsSpecified('bounding_polygon'):
        return request
    bounding_polygon_message = []
    for bounding_polygon in args.bounding_polygon:
        bounding_polygon_message.append(_PrepareBoundingPolygonMessage(bounding_polygon))
    request.referenceImage.boundingPolys = bounding_polygon_message
    return request