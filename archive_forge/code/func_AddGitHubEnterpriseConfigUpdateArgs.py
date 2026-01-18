from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitHubEnterpriseConfigUpdateArgs(parser):
    """Sets up all the argparse flags for updating a GitHub Enterprise Config.

  Args:
    parser: An argparse.ArgumentParser-like object.

  Returns:
    The parser argument with GitHub Enterprise Config flags added in.
  """
    build_flags.AddRegionFlag(parser)
    return AddGitHubEnterpriseConfigArgs(parser, update=True)