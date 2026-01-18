from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseBuildConfigArgs(trigger, args, messages, default_image, need_repo=False):
    """Parses build-config flags.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
    default_image: The docker image to use if args.dockerfile_image is empty.
    need_repo: Whether or not a repo needs to be included explicitly in flags.
  """
    if args.build_config:
        if not need_repo:
            trigger.filename = args.build_config
        trigger.substitutions = cloudbuild_util.EncodeTriggerSubstitutions(args.substitutions, messages.BuildTrigger.SubstitutionsValue)
    if args.dockerfile:
        if args.substitutions:
            raise c_exceptions.ConflictingArgumentsException('Dockerfile and substitutions', 'Substitutions are not supported with a Dockerfile configuration.')
        image = args.dockerfile_image or default_image
        trigger.build = messages.Build(steps=[messages.BuildStep(name='gcr.io/cloud-builders/docker', dir=args.dockerfile_dir, args=['build', '-t', image, '-f', args.dockerfile, '.'])])
    if args.inline_config:
        trigger.build = cloudbuild_util.LoadMessageFromPath(args.inline_config, messages.Build, 'inline build config', ['substitutions'])
        trigger.substitutions = cloudbuild_util.EncodeTriggerSubstitutions(args.substitutions, messages.BuildTrigger.SubstitutionsValue)
    if need_repo:
        required = args.build_config or args.dockerfile
        ParseGitRepoSource(trigger, args, messages, required=required)