from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.run import run_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import sources
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _SubmitBuild(tracker, release_track, region, submit_build_request):
    """Calls Build API to submit a build."""
    run_client = run_util.GetClientInstance(release_track)
    build_messages = cloudbuild_util.GetMessagesModule()
    build_response = run_client.projects_locations_builds.Submit(submit_build_request)
    build_op = build_response.buildOperation
    json = encoding.MessageToJson(build_op.metadata)
    build = encoding.JsonToMessage(build_messages.BuildOperationMetadata, json).build
    name = f'projects/{build.projectId}/locations/{region}/operations/{build.id}'
    build_op_ref = resources.REGISTRY.ParseRelativeName(name, collection='cloudbuild.projects.locations.operations')
    build_log_url = build.logUrl
    tracker.StartStage(stages.BUILD_READY)
    tracker.UpdateHeaderMessage('Building Container.')
    tracker.UpdateStage(stages.BUILD_READY, 'Logs are available at [{build_log_url}].'.format(build_log_url=build_log_url))
    response_dict = _PollUntilBuildCompletes(build_op_ref)
    return (response_dict, build_log_url)