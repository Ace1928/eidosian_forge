from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def _AddPortRelatedUpdateArgs(parser, use_port_name=True):
    """Adds parser update subcommand arguments --port and --port-name."""
    port_group = parser.add_group(help='These flags configure the port that the health check monitors.%s' % (' If both `--port` and `--port-name` are specified, `--port` takes precedence.' if use_port_name else ''))
    port_group.add_argument('--port', type=int, help='The TCP port number that this health check monitors.')
    if use_port_name:
        port_group.add_argument('--port-name', help='        The port name that this health check monitors. By default, this is\n        empty. Setting this to an empty string will clear any existing\n        port-name value.\n        ')
    _AddUseServingPortFlag(port_group, use_port_name)