from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseRepoEventArgs(trigger, args):
    """Parses repo event related flags.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
  """
    if args.included_files:
        trigger.includedFiles = args.included_files
    if args.ignored_files:
        trigger.ignoredFiles = args.ignored_files