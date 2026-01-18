from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.future_reservations import flags
from googlecloudsdk.command_lib.compute.future_reservations import resource_args
from googlecloudsdk.command_lib.compute.future_reservations import util
def _RunCreate(compute_api, args):
    """Common routine for creating future reservation."""
    resources = compute_api.resources
    future_reservation_ref = resource_args.GetFutureReservationResourceArg().ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(compute_api.client))
    messages = compute_api.client.messages
    project = future_reservation_ref.project
    create_request = _MakeCreateRequest(args, messages, resources, project, future_reservation_ref)
    service = compute_api.client.apitools_client.futureReservations
    return compute_api.client.MakeRequests([(service, 'Insert', create_request)])