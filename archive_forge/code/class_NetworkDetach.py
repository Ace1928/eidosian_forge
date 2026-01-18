import argparse
from contextlib import closing
import io
import os
from oslo_log import log as logging
import tarfile
import time
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
from zunclient.i18n import _
class NetworkDetach(command.Command):
    """Detach neutron network from specified container."""
    log = logging.getLogger(__name__ + '.NetworkDetach')

    def get_parser(self, prog_name):
        parser = super(NetworkDetach, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to detach network.')
        network_port_args = parser.add_mutually_exclusive_group()
        network_port_args.add_argument('--network', metavar='<network>', help='The network for specified container to detach.')
        network_port_args.add_argument('--port', metavar='<port>', help='The port for specified container to detach.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['container'] = parsed_args.container
        opts['network'] = parsed_args.network
        opts['port'] = parsed_args.port
        opts = zun_utils.remove_null_parms(**opts)
        try:
            client.containers.network_detach(**opts)
            print('Request to detach network for container %s has been accepted.' % parsed_args.container)
        except Exception as e:
            print('Detach network for container %(container)s failed: %(e)s' % {'container': parsed_args.container, 'e': e})