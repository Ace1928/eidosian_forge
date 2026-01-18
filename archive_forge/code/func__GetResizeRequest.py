from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import flags as r_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
def _GetResizeRequest(args, reservation_ref, holder):
    """Create Update Request for vm_count.

  Returns:
  resize request.
  Args:
   args: The arguments given to the update command.
   reservation_ref: reservation refrence.
   holder: base_classes.ComputeApiHolder.
  """
    messages = holder.client.messages
    vm_count = None
    if args.IsSpecified('vm_count'):
        vm_count = args.vm_count
    r_resize_request = messages.ComputeReservationsResizeRequest(reservation=reservation_ref.Name(), reservationsResizeRequest=messages.ReservationsResizeRequest(specificSkuCount=vm_count), project=reservation_ref.project, zone=reservation_ref.zone)
    return r_resize_request