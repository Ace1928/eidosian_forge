from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeFutureReservationMessage(messages, reservation_name, sku_properties, time_window, share_settings, planning_status, enable_auto_delete_reservations=None, auto_created_reservations_delete_time=None, auto_created_reservations_duration=None, require_specific_reservation=None):
    """Constructs a future reservation message object."""
    future_reservation_message = messages.FutureReservation(name=reservation_name, specificSkuProperties=sku_properties, timeWindow=time_window, planningStatus=planning_status)
    if share_settings:
        future_reservation_message.shareSettings = share_settings
    if enable_auto_delete_reservations is not None:
        future_reservation_message.autoDeleteAutoCreatedReservations = enable_auto_delete_reservations
    if auto_created_reservations_delete_time is not None:
        future_reservation_message.autoCreatedReservationsDeleteTime = times.FormatDateTime(auto_created_reservations_delete_time)
    if auto_created_reservations_duration is not None:
        future_reservation_message.autoCreatedReservationsDuration = messages.Duration(seconds=auto_created_reservations_duration)
    if require_specific_reservation is not None:
        future_reservation_message.specificReservationRequired = require_specific_reservation
    return future_reservation_message