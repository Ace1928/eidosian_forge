from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ApplyFailoverPolicyArgs(messages, args, backend_service, support_failover):
    """Applies the FailoverPolicy arguments to the specified backend service.

  If there are no arguments related to FailoverPolicy, the backend service
  remains unmodified.

  Args:
    messages: The available API proto messages.
    args: The arguments passed to the gcloud command.
    backend_service: The backend service proto message object.
    support_failover: Indicates whether failover functionality is supported.
  """
    if support_failover and HasFailoverPolicyArgs(args):
        failover_policy = backend_service.failoverPolicy if backend_service.failoverPolicy else messages.BackendServiceFailoverPolicy()
        if args.connection_drain_on_failover is not None:
            failover_policy.disableConnectionDrainOnFailover = not args.connection_drain_on_failover
        if args.drop_traffic_if_unhealthy is not None:
            failover_policy.dropTrafficIfUnhealthy = args.drop_traffic_if_unhealthy
        if args.failover_ratio is not None:
            failover_policy.failoverRatio = args.failover_ratio
        backend_service.failoverPolicy = failover_policy