from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def MakeUpdateReservations(args, messages, resources):
    if args.IsSpecified('reservations_from_file'):
        return _MakeReservationsFromFile(messages, args, resources)
    elif args.IsSpecified('source_reservation'):
        return MakeSourceDestReservations(args, messages)
    else:
        return []