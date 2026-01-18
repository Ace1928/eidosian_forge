from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from six.moves import map  # pylint: disable=redefined-builtin
from the device. This flag can be specified multiple times to add multiple
def AddDeviceFlagsToParser(parser, default_for_blocked_flags=True):
    """Add flags for device commands to parser.

  Args:
    parser: argparse parser to which to add these flags.
    default_for_blocked_flags: bool, whether to populate default values for
        device blocked state flags.
  """
    for f in _GetDeviceFlags(default_for_blocked_flags):
        f.AddToParser(parser)