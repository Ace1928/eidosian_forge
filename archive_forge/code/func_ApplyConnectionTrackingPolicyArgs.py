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
def ApplyConnectionTrackingPolicyArgs(client, args, backend_service):
    """Applies the connection tracking policy arguments to the specified backend service.

  If there are no arguments related to connection tracking policy, the backend
  service remains unmodified.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_service: The backend service object.
  """
    if backend_service.connectionTrackingPolicy is not None:
        connection_tracking_policy = encoding.CopyProtoMessage(backend_service.connectionTrackingPolicy)
    else:
        connection_tracking_policy = client.messages.BackendServiceConnectionTrackingPolicy()
    if args.connection_persistence_on_unhealthy_backends:
        connection_tracking_policy.connectionPersistenceOnUnhealthyBackends = client.messages.BackendServiceConnectionTrackingPolicy.ConnectionPersistenceOnUnhealthyBackendsValueValuesEnum(args.connection_persistence_on_unhealthy_backends)
    if args.tracking_mode:
        connection_tracking_policy.trackingMode = client.messages.BackendServiceConnectionTrackingPolicy.TrackingModeValueValuesEnum(args.tracking_mode)
    if args.idle_timeout_sec:
        connection_tracking_policy.idleTimeoutSec = args.idle_timeout_sec
    if args.enable_strong_affinity:
        connection_tracking_policy.enableStrongAffinity = args.enable_strong_affinity
    if connection_tracking_policy != client.messages.BackendServiceConnectionTrackingPolicy():
        backend_service.connectionTrackingPolicy = connection_tracking_policy