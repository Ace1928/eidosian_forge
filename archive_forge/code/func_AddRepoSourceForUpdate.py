from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddRepoSourceForUpdate(flag_config):
    """Adds additional argparse flags to a group for git repo source options for update commands.

  Args:
    flag_config: Argparse argument group. Git repo source flags will be added to
      this group.
  """
    source_to_build = flag_config.add_mutually_exclusive_group()
    source_to_build.add_argument('--source-to-build-repository', hidden=True, help='Repository resource (2nd gen) to use, in the format "projects/*/locations/*/connections/*/repositories/*".\n')
    v1_gen_source = source_to_build.add_argument_group()
    v1_gen_source.add_argument('--source-to-build-github-enterprise-config', help='The resource name of the GitHub Enterprise config that should be applied to\nthis source (1st gen).\nFormat: projects/{project}/locations/{location}/githubEnterpriseConfigs/{id}\nor projects/{project}/githubEnterpriseConfigs/{id}\n')
    v1_gen_repo_info = v1_gen_source.add_argument_group()
    v1_gen_repo_info.add_argument('--source-to-build-repo-type', required=True, help='Type of the repository (1st gen). Currently only GitHub and Cloud Source\nRepository types are supported.\n')
    v1_gen_repo_info.add_argument('--source-to-build-uri', required=True, help='The URI of the repository that should be applied to this source (1st gen).\n')
    ref_config = flag_config.add_mutually_exclusive_group()
    ref_config.add_argument('--source-to-build-branch', help='Branch to build.')
    ref_config.add_argument('--source-to-build-tag', help='Tag to build.')