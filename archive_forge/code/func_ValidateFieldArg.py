from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def ValidateFieldArg(ref, unused_args, request):
    """Python hook to validate that the field reference is correctly specified.

  The user should be able to describe database-wide settings as well as
  collection-group wide settings; however it doesn't make sense to describe a
  particular field path's settings unless the collection group was also
  specified. The API will catch this but it's better to do it here for a clearer
  error message.

  Args:
    ref: The field resource reference.
    unused_args: The parsed arg namespace (unused).
    request: The field describe request.
  Returns:
    The original request assuming the field configuration is valid.
  Raises:
    InvalidArgumentException: If the field resource is invalid.
  """
    if ref.fieldsId != GetDefaultFieldPathFallthrough() and ref.collectionGroupsId == GetDefaultFieldCollectionGroupFallthrough():
        raise exceptions.InvalidArgumentException('FIELD', 'Collection group must be provided if the field path was specified.')
    return request