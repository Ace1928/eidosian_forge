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
def CreateImage(tracker, build_image, build_source, build_pack, release_track, already_activated_services, region: str, resource_ref, delegate_builds=False, base_image=None):
    """Creates an image from Source."""
    delegate_builds = base_image or delegate_builds
    if release_track is base.ReleaseTrack.ALPHA and delegate_builds:
        submit_build_request = _PrepareSubmitBuildRequest(tracker, build_image, build_source, build_pack, region, resource_ref, release_track, base_image)
        response_dict, build_log_url = _SubmitBuild(tracker, release_track, region, submit_build_request)
    else:
        build_messages, build_config = _PrepareBuildConfig(tracker, build_image, build_source, build_pack, release_track, region, resource_ref)
        response_dict, build_log_url = _BuildFromSource(tracker, build_messages, build_config, skip_activation_prompt=already_activated_services)
    if response_dict and response_dict['status'] != 'SUCCESS':
        tracker.FailStage(stages.BUILD_READY, None, message='Container build failed and logs are available at [{build_log_url}].'.format(build_log_url=build_log_url))
        return None
    else:
        tracker.CompleteStage(stages.BUILD_READY)
        return response_dict['results']['images'][0]['digest']