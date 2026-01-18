from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddBitbucketServerConfigCreateArgs(parser):
    """Set up all the argparse flags for creating a Bitbucket Server Config.

  Args:
    parser: An argparse.ArgumentParser-like object.

  Returns:
    The parser argument with Bitbucket Server Config flags added in.
  """
    return AddBitbucketServerConfigArgs(parser, update=False)