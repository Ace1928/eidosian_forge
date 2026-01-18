from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def _MakeSingleReservation(args, messages, holder):
    """Makes one Allocation message object."""
    reservation_ref = resource_args.GetReservationResourceArg(positional=False).ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
    return util.MakeReservationMessageFromArgs(messages, args, reservation_ref, holder.resources)