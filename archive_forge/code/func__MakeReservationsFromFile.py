from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.compute.reservations import util
from googlecloudsdk.core import yaml
def _MakeReservationsFromFile(messages, args, resources):
    reservations_yaml = yaml.load(args.reservations_from_file)
    return _ConvertYAMLToMessage(messages, reservations_yaml, resources)