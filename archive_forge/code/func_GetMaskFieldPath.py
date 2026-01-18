from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
def GetMaskFieldPath(method):
    """Gets the dotted path of mask in the api method.

  Args:
    method: APIMethod, The method specification.

  Returns:
    String or None.
  """
    possible_mask_fields = ('updateMask', 'fieldMask')
    message = method.GetRequestType()()
    for mask in possible_mask_fields:
        if hasattr(message, mask):
            return mask
    if method.request_field:
        request_field = method.request_field
        request_message = None
        if hasattr(message, request_field):
            request_message = arg_utils.GetFieldFromMessage(message, request_field).type
        for mask in possible_mask_fields:
            if hasattr(request_message, mask):
                return '{}.{}'.format(request_field, mask)
    return None