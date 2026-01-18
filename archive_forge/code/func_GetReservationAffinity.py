from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def GetReservationAffinity(args, client):
    """Returns the message of reservation affinity for the instance."""
    if args.IsSpecified('reservation_affinity'):
        type_msgs = client.messages.ReservationAffinity.ConsumeReservationTypeValueValuesEnum
        reservation_key = None
        reservation_values = []
        if args.reservation_affinity == 'none':
            reservation_type = type_msgs.NO_RESERVATION
        elif args.reservation_affinity == 'specific':
            reservation_type = type_msgs.SPECIFIC_RESERVATION
            reservation_key = RESERVATION_AFFINITY_KEY
            reservation_values = [args.reservation]
        else:
            reservation_type = type_msgs.ANY_RESERVATION
        return client.messages.ReservationAffinity(consumeReservationType=reservation_type, key=reservation_key or None, values=reservation_values)
    return None