from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import trigger_config as trigger_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class CreateGitHub(base.CreateCommand):
    """Create a build trigger for a GitHub repository."""
    detailed_help = {'EXAMPLES': '            To create a push trigger with a 1st-gen repository for all branches:\n\n              $ {command} --name="my-trigger" --service-account="projects/my-project/serviceAccounts/my-byosa@my-project.iam.gserviceaccount.com" --repo-owner="GoogleCloudPlatform" --repo-name="cloud-builders" --branch-pattern=".*" --build-config="cloudbuild.yaml"\n\n            To create a pull request trigger with a 1st-gen repository for master:\n\n              $ {command} --name="my-trigger" --service-account="projects/my-project/serviceAccounts/my-byosa@my-project.iam.gserviceaccount.com" --repo-owner="GoogleCloudPlatform" --repo-name="cloud-builders" --pull-request-pattern="^master$" --build-config="cloudbuild.yaml"\n\n            To create a pull request trigger with a 2nd gen repository for master:\n\n              $ {command} --name="my-trigger"  --repository=projects/my-project/locations/us-central1/connections/my-conn/repositories/my-repo --pull-request-pattern="^master$" --build-config="cloudbuild.yaml" --region=us-central1\n\n          '}

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        flag_config = trigger_utils.AddTriggerArgs(parser)
        gen_config = flag_config.add_mutually_exclusive_group(required=True)
        gen_config.add_argument('--repository', help='Repository resource (2nd gen) to use, in the format\n"projects/*/locations/*/connections/*/repositories/*".\n')
        v1_config = gen_config.add_argument_group(help='1st-gen repository settings.')
        v1_config.add_argument('--repo-owner', help='Owner of the GitHub Repository (1st gen).', required=True)
        v1_config.add_argument('--repo-name', help='Name of the GitHub Repository (1st gen).', required=True)
        v1_config.add_argument('--enterprise-config', help='Resource name of the GitHub Enterprise config that should be applied to this\ninstallation.\n\nFor example: "projects/{$project_id}/locations/{$location_id}/githubEnterpriseConfigs/{$config_id}\n        ')
        ref_config = flag_config.add_mutually_exclusive_group(required=True)
        trigger_utils.AddBranchPattern(ref_config)
        trigger_utils.AddTagPattern(ref_config)
        pr_config = ref_config.add_argument_group(help='Pull Request settings')
        pr_config.add_argument('--pull-request-pattern', metavar='REGEX', required=True, help='A regular expression specifying which base git branch to match for\npull request events.\n\nThis pattern is used as a regex search for the base branch (the branch you are\ntrying to merge into) for pull request updates.\nFor example, --pull-request-pattern=foo will match "foo", "foobar", and "barfoo".\n\nThe syntax of the regular expressions accepted is the syntax accepted by\nRE2 and described at https://github.com/google/re2/wiki/Syntax.\n')
        trigger_utils.AddCommentControlArg(pr_config)
        trigger_utils.AddBuildConfigArgs(flag_config)
        trigger_utils.AddRepoEventArgs(flag_config)
        trigger_utils.AddIncludeLogsArgs(flag_config)

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

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        client = cloudbuild_util.GetClientInstance()
        messages = cloudbuild_util.GetMessagesModule()
        trigger = self.ParseTriggerFromFlags(args)
        project = properties.VALUES.core.project.Get(required=True)
        regionprop = properties.VALUES.builds.region.Get()
        location = args.region or regionprop or cloudbuild_util.DEFAULT_REGION
        parent = resources.REGISTRY.Create(collection='cloudbuild.projects.locations', projectsId=project, locationsId=location).RelativeName()
        created_trigger = client.projects_locations_triggers.Create(messages.CloudbuildProjectsLocationsTriggersCreateRequest(parent=parent, buildTrigger=trigger))
        trigger_resource = resources.REGISTRY.Parse(None, collection='cloudbuild.projects.locations.triggers', api_version='v1', params={'projectsId': project, 'locationsId': location, 'triggersId': created_trigger.id})
        log.CreatedResource(trigger_resource)
        return created_trigger