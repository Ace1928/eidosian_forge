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
def PrepareProductLabelsForProductCreationRequest(ref, args, request):
    """Sets labels if user specifies the --product-labels in product creation."""
    del ref
    if not args.IsSpecified('product_labels'):
        return request
    else:
        labels = _FormatLabelsArgsToKeyValuePairs(args.product_labels)
        request.product.productLabels = _FormatKeyValuePairsToLabelsMessage(labels)
        return request