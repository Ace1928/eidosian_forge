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
def FixProductInRemoveProductFromProductSetRequest(ref, args, request):
    """Sets product field to the full name of the product."""
    product_name = _GetProductFullName(ref, args)
    request.removeProductFromProductSetRequest.product = product_name
    return request