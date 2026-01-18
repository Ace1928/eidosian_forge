from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def ValidateFieldIndexArgs(args):
    """Validates the repeated --index arg.

  Args:
    args: The parsed arg namespace.
  Raises:
    InvalidArgumentException: If the provided indexes are incorrectly specified.
  """
    if not args.IsSpecified('index'):
        return
    for index in args.index:
        for field in index.fields:
            order = field.order
            array_config = field.arrayConfig
            if order and array_config or (not order and (not array_config)):
                raise exceptions.InvalidArgumentException('--index', "Exactly one of 'order' or 'array-config' must be specified for each --index flag provided.")