from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def AddAutoDeleteFlags(parser, is_update=False):
    """Adds all flags needed for the modifying the auto-delete properties."""
    GetAutoDeleteAutoCreatedReservationsFlag(required=False if is_update else True).AddToParser(parser)
    auto_delete_time_settings_group = base.ArgumentGroup('Manage the auto-delete time properties.', required=False, mutex=True)
    auto_delete_time_settings_group.AddArgument(GetAutoCreatedReservationsDeleteTimeFlag())
    auto_delete_time_settings_group.AddArgument(GetAutoCreatedReservationsDurationFlag())
    auto_delete_time_settings_group.AddToParser(parser)