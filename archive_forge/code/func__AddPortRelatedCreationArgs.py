from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def _AddPortRelatedCreationArgs(parser, use_port_name=True, use_serving_port=True, port_type='TCP', default_port=80):
    """Adds parser create subcommand arguments --port and --port-name."""
    port_group_help = ['These flags configure the port that the health check monitors.']
    if default_port:
        port_group_help.append('If none is specified, the default port of 80 is used.')
    if use_port_name:
        port_group_help.append('If both `--port` and `--port-name` are specified, `--port` takes precedence.')
    port_group = parser.add_group(help=' '.join(port_group_help))
    port_group.add_argument('--port', type=int, default=default_port, help='      The {} port number that this health check monitors.\n      '.format(port_type))
    if use_port_name:
        port_group.add_argument('--port-name', help='        The port name that this health check monitors. By default, this is\n        empty.\n        ')
    if use_serving_port:
        _AddUseServingPortFlag(port_group, use_port_name)