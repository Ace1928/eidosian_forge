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
class KillContainer(command.Command):
    """Kill one or more running container(s)"""
    log = logging.getLogger(__name__ + '.KillContainers')

    def get_parser(self, prog_name):
        parser = super(KillContainer, self).get_parser(prog_name)
        parser.add_argument('containers', metavar='<container>', nargs='+', help='ID or name of the (container)s to kill.')
        parser.add_argument('--signal', metavar='<signal>', default=None, help='The signal to kill')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        for container in parsed_args.containers:
            opts = {}
            opts['id'] = container
            opts['signal'] = parsed_args.signal
            opts = zun_utils.remove_null_parms(**opts)
            try:
                client.containers.kill(**opts)
                print(_('Request to send kill signal to container %s has been accepted') % container)
            except Exception as e:
                print('kill signal for container %(container)s failed: %(e)s' % {'container': container, 'e': e})