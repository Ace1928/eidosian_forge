from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def UpdateRequestWithConditionFromFile(ref, args, request):
    """Python hook to add condition from --condition-from-file to request.

  Args:
    ref: A resource ref to the parsed resource.
    args: Parsed args namespace.
    request: The apitools request message to be modified.

  Returns:
    The modified apitools request message.
  """
    del ref
    if args.IsSpecified('condition_from_file'):
        _, messages = util.GetClientAndMessages()
        condition_message = messages.Expr(description=args.condition_from_file.get('description'), title=args.condition_from_file.get('title'), expression=args.condition_from_file.get('expression'))
        request.condition = condition_message
    return request