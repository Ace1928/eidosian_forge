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
def UpdateLabelsAndUpdateMaskForProductUpdateRequest(product_ref, args, patch_request):
    """Updates product labels field."""
    if not args.IsSpecified('add_product_labels') and (not args.IsSpecified('remove_product_labels')) and (not args.IsSpecified('clear_product_labels')):
        return patch_request
    existing_labels = _GetExistingProductLabels(product_ref)
    existing_labels = _ExtractKeyValuePairsFromLabelsMessage(existing_labels)
    existing_labels_copy = copy.deepcopy(existing_labels)
    if args.clear_product_labels:
        existing_labels = _ClearLabels(existing_labels)
    if args.remove_product_labels:
        labels_to_remove = _FormatLabelsArgsToKeyValuePairs(args.remove_product_labels)
        existing_labels = _RemoveLabels(existing_labels, labels_to_remove)
    if args.add_product_labels:
        labels_to_add = _FormatLabelsArgsToKeyValuePairs(args.add_product_labels)
        existing_labels = _AddLabels(existing_labels, labels_to_add)
    if _LabelsUpdated(existing_labels, existing_labels_copy):
        patch_request = _AddFieldToUpdateMask('productLabels', patch_request)
        updated_labels_message = _FormatKeyValuePairsToLabelsMessage(existing_labels)
        if patch_request.product is None:
            patch_request.product = api_utils.GetMessage().Product()
        patch_request.product.productLabels = updated_labels_message
    return patch_request