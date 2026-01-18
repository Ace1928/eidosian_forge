from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitHubEnterpriseConfigCreateArgs(parser):
    """Sets up all the argparse flags for creating a GitHub Enterprise Config.

  Args:
    parser: An argparse.ArgumentParser-like object.

  Returns:
    The parser argument with GitHub Enterprise Config flags added in.
  """
    region = parser.add_argument_group()
    region.add_argument('--name', required=True, help='The name of the GitHub Enterprise config.')
    region.add_argument('--region', required=True, help='The region of the Cloud Build Service to use.\nMust be set to a supported region\nname (e.g. `us-central1`).\nIf unset, `builds/region`, which is the default\nregion to use when working with Cloud Build resources, is used. If builds/region\nis unset, region is set to `global`.\n')
    return AddGitHubEnterpriseConfigArgs(parser, update=False)