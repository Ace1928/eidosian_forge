from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddClearableArgument(parser, for_create, arg_name, arg_type, arg_help, clear_help, choices=None):
    """Adds an argument for a field that can be cleared in an update."""
    if for_create:
        parser.add_argument('--{}'.format(arg_name), type=arg_type, help=arg_help, choices=choices)
    else:
        mutex = parser.add_mutually_exclusive_group(required=False)
        mutex.add_argument('--{}'.format(arg_name), type=arg_type, help=arg_help, choices=choices)
        mutex.add_argument('--clear-{}'.format(arg_name), action='store_true', default=False, help=clear_help)