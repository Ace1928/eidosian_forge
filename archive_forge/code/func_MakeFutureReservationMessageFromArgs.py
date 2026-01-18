from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakeFutureReservationMessageFromArgs(messages, resources, args, future_reservation_ref):
    """Construct future reservation message from args passed in."""
    local_ssds = reservation_util.MakeLocalSsds(messages, getattr(args, 'local_ssd', None))
    accelerators = reservation_util.MakeGuestAccelerators(messages, getattr(args, 'accelerator', None))
    allocated_instance_properties = MakeAllocatedInstanceProperties(messages, args.machine_type, args.min_cpu_platform, local_ssds, accelerators, getattr(args, 'location_hint', None), getattr(args, 'maintenance_freeze_duration', None), getattr(args, 'maintenance_interval', None))
    source_instance_template_ref = reservation_util.ResolveSourceInstanceTemplate(args, resources) if getattr(args, 'source_instance_template', None) else None
    sku_properties = MakeSpecificSKUPropertiesMessage(messages, allocated_instance_properties, args.total_count, source_instance_template_ref)
    time_window = MakeTimeWindowMessage(messages, args.start_time, getattr(args, 'end_time', None), getattr(args, 'duration', None))
    share_settings = MakeShareSettings(messages, args, getattr(args, 'share_setting', None))
    planning_status = MakePlanningStatus(messages, getattr(args, 'planning_status', None))
    enable_auto_delete_reservations = None
    if args.IsSpecified('auto_delete_auto_created_reservations'):
        enable_auto_delete_reservations = getattr(args, 'auto_delete_auto_created_reservations')
    auto_created_reservations_delete_time = None
    if args.IsSpecified('auto_created_reservations_delete_time'):
        auto_created_reservations_delete_time = getattr(args, 'auto_created_reservations_delete_time')
    auto_created_reservations_duration = None
    if args.IsSpecified('auto_created_reservations_duration'):
        auto_created_reservations_duration = getattr(args, 'auto_created_reservations_duration')
    require_specific_reservation = getattr(args, 'require_specific_reservation', None)
    return MakeFutureReservationMessage(messages, future_reservation_ref.Name(), sku_properties, time_window, share_settings, planning_status, enable_auto_delete_reservations, auto_created_reservations_delete_time, auto_created_reservations_duration, require_specific_reservation)