from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddReservationAffinityGroup(parser, group_text, affinity_text, support_specific_then_x_affinity):
    """Adds the argument group to handle reservation affinity configurations."""
    group = parser.add_group(help=group_text)
    choices = {'any': 'Consume any available, matching reservation.', 'none': 'Do not consume from any reserved capacity.', 'specific': 'Must consume from a specific reservation.'}
    if support_specific_then_x_affinity:
        choices.update({'specific-then-any-reservation': 'Prefer to consume from a specific reservation, but still consume any available matching reservation if the specified reservation is not available or exhausted.', 'specific-then-no-reservation': 'Prefer to consume from a specific reservation, but still consume from the on-demand pool if the specified reservation is not available or exhausted.'})
    group.add_argument('--reservation-affinity', choices=choices, default='any', help=affinity_text)
    if support_specific_then_x_affinity:
        reservation_help_text = '\nThe name of the reservation, required when `--reservation-affinity` is one of: `specific`, `specific-then-any-reservation` or `specific-then-no-reservation`.\n'
    else:
        reservation_help_text = '\nThe name of the reservation, required when `--reservation-affinity=specific`.\n'
    group.add_argument('--reservation', help=reservation_help_text)