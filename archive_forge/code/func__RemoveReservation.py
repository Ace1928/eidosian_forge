from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.api_lib.bms.bms_client import IpRangeReservation
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import exceptions
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _RemoveReservation(reservations, remove_key_dict):
    _ValidateAgainstSpec(remove_key_dict, flags.IP_RESERVATION_KEY_SPEC, 'remove-ip-range-reservation')
    start_address = remove_key_dict['start-address']
    end_address = remove_key_dict['end-address']
    for i, res in enumerate(reservations):
        if res.start_address == start_address and res.end_address == end_address:
            return reservations[:i] + reservations[i + 1:]
    raise LookupError('Cannot find an IP range reservation with start-address [{}] and end-address [{}]'.format(start_address, end_address))