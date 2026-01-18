from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitRepoSource(flag_config):
    """Adds additional argparse flags to a group for git repo source options.

  Args:
    flag_config: argparse argument group. Git repo source flags will be added to
      this group.
  """
    repo_config = flag_config.add_argument_group(help='Flags for repository and branch information')
    gen_config = repo_config.add_mutually_exclusive_group(help='Flags for repository information')
    gen_config.add_argument('--repository', help='Repository resource (2nd gen) to use, in the format "projects/*/locations/*/connections/*/repositories/*".\n')
    v1_repo = gen_config.add_argument_group(help='1st-gen repository settings.')
    v1_repo.add_argument('--repo', required=True, help='URI of the repository (1st gen). Currently only HTTP URIs for GitHub and Cloud\nSource Repositories are supported.\n')
    v1_repo.add_argument('--repo-type', required=True, help='Type of the repository (1st gen). Currently only GitHub and Cloud Source Repository types\nare supported.\n')
    config = v1_repo.add_mutually_exclusive_group()
    config.add_argument('--github-enterprise-config', help='The resource name of the GitHub Enterprise config that should be applied to this source (1st gen).\nFormat: projects/{project}/locations/{location}/githubEnterpriseConfigs/{id} or projects/{project}/githubEnterpriseConfigs/{id}\n')
    config.add_argument('--bitbucket-server-config', hidden=True, help='The resource name of the Bitbucket Server config that should be applied to this source (1st gen).\nFormat: projects/{project}/locations/{location}/bitbucketServerConfigs/{id}\n')
    ref_config = repo_config.add_mutually_exclusive_group()
    ref_config.add_argument('--branch', help='Branch to build.')
    ref_config.add_argument('--tag', help='Tag to build.')