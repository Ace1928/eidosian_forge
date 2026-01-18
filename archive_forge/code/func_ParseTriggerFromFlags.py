from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import trigger_config as trigger_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseTriggerFromFlags(self, args):
    """Parses command line arguments into a build trigger.

    Args:
      args: An argparse arguments object.

    Returns:
      A build trigger object.

    Raises:
      RequiredArgumentException: If comment_control is defined but
      pull_request_pattern isn't.
    """
    project = properties.VALUES.core.project.Get(required=True)
    messages = cloudbuild_util.GetMessagesModule()
    trigger, done = trigger_utils.ParseTriggerArgs(args, messages)
    if done:
        return trigger
    if args.repo_owner and args.repo_name:
        trigger.github = messages.GitHubEventsConfig(owner=args.repo_owner, name=args.repo_name, enterpriseConfigResourceName=args.enterprise_config)
        rcfg = trigger.github
    else:
        trigger.repositoryEventConfig = messages.RepositoryEventConfig(repository=args.repository)
        rcfg = trigger.repositoryEventConfig
    if args.pull_request_pattern:
        rcfg.pullRequest = messages.PullRequestFilter(branch=args.pull_request_pattern)
        if args.comment_control:
            rcfg.pullRequest.commentControl = messages.PullRequestFilter.CommentControlValueValuesEnum(args.comment_control)
    else:
        rcfg.push = messages.PushFilter(branch=args.branch_pattern, tag=args.tag_pattern)
    default_image = 'gcr.io/%s/github.com/%s/%s:$COMMIT_SHA' % (project, args.repo_owner, args.repo_name)
    trigger_utils.ParseBuildConfigArgs(trigger, args, messages, default_image)
    trigger_utils.ParseRepoEventArgs(trigger, args)
    trigger_utils.ParseIncludeLogsWithStatus(trigger, args, messages)
    return trigger