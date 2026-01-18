from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import release
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import promote_util
from googlecloudsdk.command_lib.deploy import release_util
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Promote(base.CreateCommand):
    """Promotes a release from one target (source), to another (destination).

  If to-target is not specified the command promotes the release from the target
  that is farthest along in the promotion sequence to its next stage in the
  promotion sequence.
  """
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        _CommonArgs(parser)

    @gcloud_exception.CatchHTTPErrorRaiseHTTPException(deploy_exceptions.HTTP_ERROR_FORMAT)
    def Run(self, args):
        release_ref = args.CONCEPTS.release.Parse()
        pipeline_ref = release_ref.Parent()
        pipeline_obj = delivery_pipeline_util.GetPipeline(pipeline_ref.RelativeName())
        failed_activity_msg = 'Cannot promote release {}.'.format(release_ref.RelativeName())
        delivery_pipeline_util.ThrowIfPipelineSuspended(pipeline_obj, failed_activity_msg)
        release_obj = release.ReleaseClient().Get(release_ref.RelativeName())
        messages = core_apis.GetMessagesModule('clouddeploy', 'v1')
        skaffold_support_state = release_util.GetSkaffoldSupportState(release_obj)
        skaffold_support_state_enum = messages.SkaffoldSupportedCondition.SkaffoldSupportStateValueValuesEnum
        if skaffold_support_state == skaffold_support_state_enum.SKAFFOLD_SUPPORT_STATE_MAINTENANCE_MODE:
            log.status.Print("WARNING: This release's Skaffold version is in maintenance mode and will be unsupported soon.\n https://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")
        if skaffold_support_state == skaffold_support_state_enum.SKAFFOLD_SUPPORT_STATE_UNSUPPORTED:
            raise core_exceptions.Error("You can't promote this release because the Skaffold version that was used to create the release is no longer supported.\nhttps://cloud.google.com/deploy/docs/using-skaffold/select-skaffold#skaffold_version_deprecation_and_maintenance_policy")
        if release_obj.abandoned:
            raise deploy_exceptions.AbandonedReleaseError('Cannot promote release.', release_ref.RelativeName())
        to_target_id = args.to_target
        if not to_target_id:
            to_target_id = promote_util.GetToTargetID(release_obj, False)
            promote_util.CheckIfInProgressRollout(release_ref, release_obj, to_target_id)
        release_util.PrintDiff(release_ref, release_obj, args.to_target)
        console_io.PromptContinue('Promoting release {} to target {}.'.format(release_ref.Name(), to_target_id), cancel_on_no=True)
        policies = deploy_policy_util.CreateDeployPolicyNamesFromIDs(pipeline_ref, args.override_deploy_policies)
        rollout_resource = promote_util.Promote(release_ref, release_obj, to_target_id, False, rollout_id=args.rollout_id, annotations=args.annotations, labels=args.labels, starting_phase_id=args.starting_phase_id, override_deploy_policies=policies)
        return rollout_resource