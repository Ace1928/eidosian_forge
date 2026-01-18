from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import rollout
from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.deploy import delivery_pipeline_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import exceptions as deploy_exceptions
from googlecloudsdk.command_lib.deploy import flags
from googlecloudsdk.command_lib.deploy import resource_args
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class IgnoreJob(base.CreateCommand):
    """Ignores a specified job and phase combination on a rollout."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddRolloutResourceArg(parser, positional=True)
        flags.AddJobId(parser)
        flags.AddPhaseId(parser)
        flags.AddOverrideDeployPolicies(parser)

    @gcloud_exception.CatchHTTPErrorRaiseHTTPException(deploy_exceptions.HTTP_ERROR_FORMAT)
    def Run(self, args):
        rollout_ref = args.CONCEPTS.rollout.Parse()
        pipeline_ref = rollout_ref.Parent().Parent()
        pipeline_obj = delivery_pipeline_util.GetPipeline(pipeline_ref.RelativeName())
        failed_activity_msg = 'Cannot ignore job on rollout {}.'.format(rollout_ref.RelativeName())
        delivery_pipeline_util.ThrowIfPipelineSuspended(pipeline_obj, failed_activity_msg)
        log.status.Print('Ignoring job {} in phase {} of rollout {}.\n'.format(args.job_id, args.phase_id, rollout_ref.RelativeName()))
        policies = deploy_policy_util.CreateDeployPolicyNamesFromIDs(pipeline_ref, args.override_deploy_policies)
        return rollout.RolloutClient().IgnoreJob(rollout_ref.RelativeName(), args.job_id, args.phase_id, override_deploy_policies=policies)