from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def _AddTcpRelatedArgsImpl(add_info_about_clearing, parser):
    """Adds TCP-related subcommand parser arguments."""
    request_help = '      An optional string of up to 1024 characters to send once the health check\n      TCP connection has been established. The health checker then looks for a\n      reply of the string provided in the `--response` field.\n\n      If `--response` is not configured, the health checker does not wait for a\n      response and regards the probe as successful if the TCP or SSL handshake\n      was successful.\n      '
    response_help = '      An optional string of up to 1024 characters that the health checker\n      expects to receive from the instance. If the response is not received\n      exactly, the health check probe fails. If `--response` is configured, but\n      not `--request`, the health checker will wait for a response anyway.\n      Unless your system automatically sends out a message in response to a\n      successful handshake, only configure `--response` to match an explicit\n      `--request`.\n      '
    if add_info_about_clearing:
        request_help += '\n      Setting this to an empty string will clear any existing request value.\n      '
        response_help += '      Setting this to an empty string will clear any existing\n      response value.\n      '
    parser.add_argument('--request', help=request_help)
    parser.add_argument('--response', help=response_help)