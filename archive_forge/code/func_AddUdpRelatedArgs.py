from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddUdpRelatedArgs(parser, request_and_response_required=True):
    """Adds parser arguments related to UDP."""
    _AddPortRelatedCreationArgs(parser, use_serving_port=False, port_type='UDP', default_port=None)
    parser.add_argument('--request', required=request_and_response_required, help='      Application data to send in payload of an UDP packet. It is an error if\n      this is empty.\n      ')
    parser.add_argument('--response', required=request_and_response_required, help='      The bytes to match against the beginning of the response data.\n      It is an error if this is empty.\n      ')