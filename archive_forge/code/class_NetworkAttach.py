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
class NetworkAttach(command.Command):
    """Attach neutron network to specified container."""
    log = logging.getLogger(__name__ + '.NetworkAttach')

    def get_parser(self, prog_name):
        parser = super(NetworkAttach, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to attach network.')
        parser.add_argument('--network', metavar='<network>', help='The network for specified container to attach.')
        parser.add_argument('--port', metavar='<port>', help='The port for specified container to attach.')
        parser.add_argument('--fixed-ip', metavar='<fixed_ip>', help='The fixed-ip that container will attach to.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['container'] = parsed_args.container
        opts['network'] = parsed_args.network
        opts['port'] = parsed_args.port
        opts['fixed_ip'] = parsed_args.fixed_ip
        opts = zun_utils.remove_null_parms(**opts)
        try:
            client.containers.network_attach(**opts)
            print('Request to attach network to container %s has been accepted.' % parsed_args.container)
        except Exception as e:
            print('Attach network to container %(container)s failed: %(e)s' % {'container': parsed_args.container, 'e': e})