from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils as mig_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateAutoscaling(base.Command):
    """Update autoscaling parameters of a managed instance group."""
    clear_scale_down = False

    @staticmethod
    def Args(parser):
        _CommonArgs(parser)
        mig_utils.AddPredictiveAutoscaling(parser, standard=False)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = instance_groups_flags.CreateGroupReference(client, holder.resources, args)
        mig_utils.GetInstanceGroupManagerOrThrow(igm_ref, client)
        old_autoscaler = mig_utils.AutoscalerForMigByRef(client, holder.resources, igm_ref)
        if mig_utils.IsAutoscalerNew(old_autoscaler):
            raise NoMatchingAutoscalerFoundError('Instance group manager [{}] has no existing autoscaler; cannot update.'.format(igm_ref.Name()))
        autoscalers_client = autoscalers_api.GetClient(client, igm_ref)
        new_autoscaler = autoscalers_client.message_type(name=old_autoscaler.name, autoscalingPolicy=client.messages.AutoscalingPolicy())
        if args.IsSpecified('mode'):
            mode = mig_utils.ParseModeString(args.mode, client.messages)
            new_autoscaler.autoscalingPolicy.mode = mode
        if args.IsSpecified('clear_scale_in_control'):
            new_autoscaler.autoscalingPolicy.scaleInControl = None
        else:
            new_autoscaler.autoscalingPolicy.scaleInControl = mig_utils.BuildScaleIn(args, client.messages)
        if self.clear_scale_down and args.IsSpecified('clear_scale_down_control'):
            new_autoscaler.autoscalingPolicy.scaleDownControl = None
        if args.IsSpecified('cpu_utilization_predictive_method'):
            cpu_predictive_enum = client.messages.AutoscalingPolicyCpuUtilization.PredictiveMethodValueValuesEnum
            new_autoscaler.autoscalingPolicy.cpuUtilization = client.messages.AutoscalingPolicyCpuUtilization()
            new_autoscaler.autoscalingPolicy.cpuUtilization.predictiveMethod = arg_utils.ChoiceToEnum(args.cpu_utilization_predictive_method, cpu_predictive_enum)
        scheduled = mig_utils.BuildSchedules(args, client.messages)
        if scheduled:
            new_autoscaler.autoscalingPolicy.scalingSchedules = scheduled
        if args.IsSpecified('min_num_replicas'):
            new_autoscaler.autoscalingPolicy.minNumReplicas = args.min_num_replicas
        if args.IsSpecified('max_num_replicas'):
            new_autoscaler.autoscalingPolicy.maxNumReplicas = args.max_num_replicas
        return self._SendPatchRequest(args, client, autoscalers_client, igm_ref, new_autoscaler)

    def _SendPatchRequest(self, args, client, autoscalers_client, igm_ref, new_autoscaler):
        if args.IsSpecified('clear_scale_in_control'):
            with client.apitools_client.IncludeFields(['autoscalingPolicy.scaleInControl']):
                return autoscalers_client.Patch(igm_ref, new_autoscaler)
        elif self.clear_scale_down and args.IsSpecified('clear_scale_down_control'):
            with client.apitools_client.IncludeFields(['autoscalingPolicy.scaleDownControl']):
                return autoscalers_client.Patch(igm_ref, new_autoscaler)
        else:
            return autoscalers_client.Patch(igm_ref, new_autoscaler)