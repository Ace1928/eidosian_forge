from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
def GetMaskString(args, spec, mask_path, is_dotted=True):
    """Gets the fieldMask that is required for update api calls.

  Args:
    args: The argparse parser.
    spec: The CommandData class.
    mask_path: string, the dotted path of mask in the api method
    is_dotted: Boolean, True if the dotted path of the name is returned.

  Returns:
    A String, represents a mask specifying which fields in the resource should
    be updated.

  Raises:
    NoFieldsSpecifiedError: this error would happen when no args are specified.
  """
    if not args.GetSpecifiedArgs():
        raise NoFieldsSpecifiedError('Must specify at least one valid parameter to update.')
    field_set = set()
    for param in _GetSpecParams(spec.arguments.params):
        field_set.update(_GetMaskFields(param, args, mask_path, is_dotted))
    return ','.join(sorted(field_set))