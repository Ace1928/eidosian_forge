from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddRepoEventArgs(flag_config):
    """Adds additional argparse flags related to repo events.

  Args:
    flag_config: argparse argument group. Additional flags will be added to this
      group to cover common build configuration settings.
  """
    flag_config.add_argument('--included-files', help='Glob filter. Changes affecting at least one included file will trigger builds.\n', type=arg_parsers.ArgList(), metavar='GLOB')
    flag_config.add_argument('--ignored-files', help="Glob filter. Changes only affecting ignored files won't trigger builds.\n", type=arg_parsers.ArgList(), metavar='GLOB')