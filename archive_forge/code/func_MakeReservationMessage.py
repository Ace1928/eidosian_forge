from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeReservationMessage(messages, reservation_name, share_settings, specific_reservation, resource_policies, require_specific_reservation, reservation_zone, delete_at_time=None, delete_after_duration=None):
    """Constructs a single reservations message object."""
    reservation_message = messages.Reservation(name=reservation_name, specificReservation=specific_reservation, specificReservationRequired=require_specific_reservation, zone=reservation_zone)
    if share_settings:
        reservation_message.shareSettings = share_settings
    if resource_policies:
        reservation_message.resourcePolicies = resource_policies
    if delete_at_time:
        reservation_message.deleteAtTime = times.FormatDateTime(delete_at_time)
    if delete_after_duration:
        reservation_message.deleteAfterDuration = messages.Duration(seconds=delete_after_duration)
    return reservation_message