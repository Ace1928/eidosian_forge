from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
def _ParseArgWithShortName(args, short_name):
    """Returns the argument value for given short_name or None if not specified.

  Args:
    args: The argument object obtained by parsing the command-line arguments
      using argparse.
    short_name: The regular name for the argument to be fetched, such as
      `access_levels`.
  """
    alt_name = 'perimeter_' + short_name
    if args.IsSpecified(short_name):
        return getattr(args, short_name, None)
    elif args.IsSpecified(alt_name):
        return getattr(args, alt_name, None)
    return None