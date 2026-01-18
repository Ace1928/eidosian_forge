from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddUpdateReservationGroup(parser):
    """Add reservation arguments to the update-reservations command."""
    parent_reservations_group = parser.add_group('Manage reservations that are attached to the commitment.', mutex=True)
    AddReservationsFromFileFlag(parent_reservations_group, custom_text="Path to a YAML file of two reservations' configuration.")
    reservations_group = parent_reservations_group.add_group('Specify source and destination reservations configuration.')
    AddReservationArguments(reservations_group)
    reservation_flags.GetAcceleratorFlag('--source-accelerator').AddToParser(reservations_group)
    reservation_flags.GetAcceleratorFlag('--dest-accelerator').AddToParser(reservations_group)
    reservation_flags.GetLocalSsdFlag('--source-local-ssd').AddToParser(reservations_group)
    reservation_flags.GetLocalSsdFlag('--dest-local-ssd').AddToParser(reservations_group)
    reservation_flags.GetSharedSettingFlag('--source-share-setting').AddToParser(reservations_group)
    reservation_flags.GetShareWithFlag('--source-share-with').AddToParser(reservations_group)
    reservation_flags.GetSharedSettingFlag('--dest-share-setting').AddToParser(reservations_group)
    reservation_flags.GetShareWithFlag('--dest-share-with').AddToParser(reservations_group)
    return parser