from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import properties
def GetLineItemChangeTypeEnum(raw_input):
    """Converts raw input to line item change type.

  Args:
    raw_input: Raw input of the line item change type.

  Returns:
    Converted line item change type.
  Raises:
    ValueError: The raw input is not recognized as a valid change type.
  """
    if raw_input == 'UPDATE':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequestModification().ChangeTypeValueValuesEnum.LINE_ITEM_CHANGE_TYPE_UPDATE
    elif raw_input == 'CANCEL':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequestModification().ChangeTypeValueValuesEnum.LINE_ITEM_CHANGE_TYPE_CANCEL
    elif raw_input == 'REVERT_CANCELLATION':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequestModification().ChangeTypeValueValuesEnum.LINE_ITEM_CHANGE_TYPE_REVERT_CANCELLATION
    else:
        raise ValueError('Unrecognized line item change type {}.'.format(raw_input))