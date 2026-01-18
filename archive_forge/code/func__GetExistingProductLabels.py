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
def _GetExistingProductLabels(product_ref):
    """Fetches the existing product labels to update."""
    get_request_message = api_utils.GetMessage().VisionProjectsLocationsProductsGetRequest(name=product_ref.RelativeName())
    product = api_utils.GetClient().projects_locations_products.Get(get_request_message)
    return product.productLabels