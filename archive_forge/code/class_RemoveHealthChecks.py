from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.http_health_checks import (
from googlecloudsdk.command_lib.compute.target_pools import flags
class RemoveHealthChecks(base.SilentCommand):
    """Remove an HTTP health check from a target pool.

  *{command}* is used to remove an HTTP health check
  from a target pool. Health checks are used to determine
  the health status of instances in the target pool. For more
  information on health checks and load balancing, see
  [](https://cloud.google.com/compute/docs/load-balancing-and-autoscaling/)
  """
    HEALTH_CHECK_ARG = None
    TARGET_POOL_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.HEALTH_CHECK_ARG = http_health_check_flags.HttpHealthCheckArgumentForTargetPool('remove from')
        cls.HEALTH_CHECK_ARG.AddArgument(parser)
        cls.TARGET_POOL_ARG = flags.TargetPoolArgument(help_suffix=' from which to remove the health check.')
        cls.TARGET_POOL_ARG.AddArgument(parser, operation_type='remove health checks from')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        http_health_check_ref = self.HEALTH_CHECK_ARG.ResolveAsResource(args, holder.resources)
        target_pool_ref = self.TARGET_POOL_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        request = client.messages.ComputeTargetPoolsRemoveHealthCheckRequest(region=target_pool_ref.region, project=target_pool_ref.project, targetPool=target_pool_ref.Name(), targetPoolsRemoveHealthCheckRequest=client.messages.TargetPoolsRemoveHealthCheckRequest(healthChecks=[client.messages.HealthCheckReference(healthCheck=http_health_check_ref.SelfLink())]))
        return client.MakeRequests([(client.apitools_client.targetPools, 'RemoveHealthCheck', request)])