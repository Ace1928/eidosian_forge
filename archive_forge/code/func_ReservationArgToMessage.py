from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def ReservationArgToMessage(reservation, accelerator, local_ssd, share_setting, share_with, args, messages):
    """Convert single reservation argument into a message."""
    accelerators = util.MakeGuestAccelerators(messages, getattr(args, accelerator, None))
    local_ssds = util.MakeLocalSsds(messages, getattr(args, local_ssd, None))
    share_settings = util.MakeShareSettingsWithArgs(messages, args, getattr(args, share_setting, None), share_with)
    reservation = getattr(args, reservation, None)
    specific_allocation = util.MakeSpecificSKUReservationMessage(messages, reservation.get('vm-count', None), accelerators, local_ssds, reservation.get('machine-type', None), reservation.get('min-cpu-platform', None))
    a_msg = util.MakeReservationMessage(messages, reservation.get('reservation', None), share_settings, specific_allocation, reservation.get('resource-policies', None), reservation.get('require-specific-reservation', None), reservation.get('reservation-zone', None))
    return a_msg