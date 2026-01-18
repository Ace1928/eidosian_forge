from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import trigger_config as trigger_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class CreateWebhook(base.CreateCommand):
    """Create a build trigger with a Webhook trigger event."""
    detailed_help = {'EXAMPLES': '            To create a Webhook trigger that requires secret `projects/my-project/secrets/my-secret/versions/2` and builds off branch `my-branch` in a GitHub repository named `my-repo`:\n\n              $ {command} --name=my-webhook-trigger --service-account="projects/my-project/serviceAccounts/my-byosa@my-project.iam.gserviceaccount.com" --secret=projects/my-project/secrets/my-secret/versions/2 --repo=https://www.github.com/owner/repo --repo-type=GITHUB --branch=my-branch\n\n            To create a Webhook trigger that requires secret `projects/my-project/secrets/my-secret/versions/2` and builds off branch `my-branch` in a 2nd-gen GitHub repository:\n\n              $ {command} --name=my-webhook-trigger --service-account="projects/my-project/serviceAccounts/my-byosa@my-project.iam.gserviceaccount.com" --secret=projects/my-project/secrets/my-secret/versions/2 --branch=my-branch --repository=projects/my-proj/locations/us-west1/connections/my-conn/repositories/my-repo\n\n          '}

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object.
    """
        flag_config = trigger_utils.AddTriggerArgs(parser)
        flag_config.add_argument('--secret', help='The full path of the secret version required to validate webhook requests\nagainst this trigger.\nFor example, projects/my-project/secrets/my-secret/versions/1.\n', required=True)
        trigger_utils.AddBuildConfigArgs(flag_config)
        trigger_utils.AddGitRepoSource(flag_config)
        trigger_utils.AddFilterArg(flag_config)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: An argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      The newly created trigger.
    """
        client = cloudbuild_util.GetClientInstance()
        messages = cloudbuild_util.GetMessagesModule()
        trigger = messages.BuildTrigger()
        if args.trigger_config:
            trigger = cloudbuild_util.LoadMessageFromPath(path=args.trigger_config, msg_type=messages.BuildTrigger, msg_friendly_name='build trigger config', skip_camel_case=['substitutions'])
        else:
            trigger = ParseTriggerFromFlags(args)
        project = properties.VALUES.core.project.Get(required=True)
        regionprop = properties.VALUES.builds.region.Get()
        location = args.region or regionprop or cloudbuild_util.DEFAULT_REGION
        parent = resources.REGISTRY.Create(collection='cloudbuild.projects.locations', projectsId=project, locationsId=location).RelativeName()
        created_trigger = client.projects_locations_triggers.Create(messages.CloudbuildProjectsLocationsTriggersCreateRequest(parent=parent, buildTrigger=trigger))
        trigger_resource = resources.REGISTRY.Parse(None, collection='cloudbuild.projects.locations.triggers', api_version='v1', params={'projectsId': project, 'locationsId': location, 'triggersId': created_trigger.id})
        log.CreatedResource(trigger_resource)
        return created_trigger