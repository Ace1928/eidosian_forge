from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseBuildConfigArgsForUpdate(trigger, old_trigger, args, messages, update_mask, default_image, has_build_config=False, has_repo_source=False, has_file_source=False):
    """Parses build-config flags for update command.

  Args:
    trigger: The trigger to populate.
    old_trigger: The existing trigger to be updated.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
    update_mask: The fields to be updated.
    default_image: The default docker image to use.
    has_build_config: Whether it is possible for the trigger to have
      filename.
    has_repo_source: Whether it is possible for the trigger to have
      source_to_build.
    has_file_source: Whether it is possible for the trigger to have
      git_file_source.
  """
    if has_build_config:
        trigger.filename = args.build_config
    if args.dockerfile:
        if args.update_substitutions:
            raise c_exceptions.ConflictingArgumentsException('Dockerfile and substitutions', 'Substitutions are not supported with a Dockerfile configuration.')
        if args.dockerfile_dir:
            dockerfile_dir = args.dockerfile_dir
        elif old_trigger.build and old_trigger.build.steps:
            dockerfile_dir = old_trigger.build.steps[0].dir
        else:
            dockerfile_dir = '/'
        dockerfile_image = args.dockerfile_image or default_image
        trigger.build = messages.Build(steps=[messages.BuildStep(name='gcr.io/cloud-builders/docker', dir=dockerfile_dir, args=['build', '-t', dockerfile_image, '-f', args.dockerfile, '.'])])
    if args.inline_config:
        trigger.build = cloudbuild_util.LoadMessageFromPath(args.inline_config, messages.Build, 'inline build config', ['substitutions'])
    if args.update_substitutions:
        trigger.substitutions = cloudbuild_util.EncodeUpdatedTriggerSubstitutions(old_trigger.substitutions, args.update_substitutions, messages)
    if args.clear_substitutions:
        trigger.substitutions = cloudbuild_util.EncodeEmptyTriggerSubstitutions(messages)
    if args.remove_substitutions:
        trigger.substitutions = cloudbuild_util.RemoveTriggerSubstitutions(old_trigger.substitutions, args.remove_substitutions, messages)
    if has_repo_source and (args.source_to_build_uri or args.source_to_build_branch or args.source_to_build_tag or args.source_to_build_repo_type or args.source_to_build_github_enterprise_config or args.source_to_build_repository):
        ParseGitRepoSourceForUpdate(trigger, args, messages, update_mask)
    if has_file_source and (args.git_file_source_uri or args.git_file_source_path or args.git_file_source_repo_type or args.git_file_source_branch or args.git_file_source_tag or args.git_file_source_github_enterprise_config or args.git_file_source_repository):
        ParseGitFileSource(trigger, args, messages, update_mask)