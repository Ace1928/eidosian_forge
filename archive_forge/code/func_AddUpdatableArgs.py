from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddUpdatableArgs(cls, parser, operation_type):
    """Adds top-level backend bucket arguments that can be updated.

  Args:
    cls: type, Class to add backend bucket argument to.
    parser: The argparse parser.
    operation_type: str, operation_type forwarded to AddArgument(...)
  """
    cls.BACKEND_BUCKET_ARG = BackendBucketArgument()
    cls.BACKEND_BUCKET_ARG.AddArgument(parser, operation_type=operation_type)
    parser.add_argument('--description', help='An optional, textual description for the backend bucket.')
    parser.add_argument('--enable-cdn', action=arg_parsers.StoreTrueFalseAction, help='      Enable Cloud CDN for the backend bucket. Cloud CDN can cache HTTP\n      responses from a backend bucket at the edge of the network, close to\n      users.')