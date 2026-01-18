from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseGitFileSource(trigger, args, messages, update_mask):
    """Parses git repo source flags.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
    update_mask: The fields to be updated.
  """
    trigger.gitFileSource = messages.GitFileSource()
    revision = None
    if args.git_file_source_branch:
        revision = 'refs/heads/' + args.git_file_source_branch
    elif args.git_file_source_tag:
        revision = 'refs/tags/' + args.git_file_source_tag
    trigger.gitFileSource.revision = revision
    if args.git_file_source_repository:
        trigger.gitFileSource.repository = args.git_file_source_repository
        update_mask.append('git_file_source.uri')
        update_mask.append('git_file_source.repo_type')
        update_mask.append('git_file_source.github_enterprise_config')
        update_mask.append('git_file_source.path')
    elif args.git_file_source_github_enterprise_config or args.git_file_source_uri or args.git_file_source_path or args.git_file_source_repo_type:
        trigger.gitFileSource.path = args.git_file_source_path
        trigger.gitFileSource.uri = args.git_file_source_uri
        trigger.gitFileSource.githubEnterpriseConfig = args.git_file_source_github_enterprise_config
        if args.git_file_source_repo_type:
            trigger.gitFileSource.repoType = messages.GitFileSource.RepoTypeValueValuesEnum(args.git_file_source_repo_type)
        update_mask.append('git_file_source.repository')