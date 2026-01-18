from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log as sdk_log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def AddBucketLocationArg(parser, required, help_string):
    """Adds a location argument.

  Args:
    parser: parser to which to add args.
    required: whether the arguments is required.
    help_string: the help string.
  """
    parser.add_argument('--location', required=required, metavar='LOCATION', type=arg_parsers.RegexpValidator('.+', 'must be non-empty'), help=help_string)