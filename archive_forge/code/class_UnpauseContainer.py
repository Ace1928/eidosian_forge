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
class UnpauseContainer(command.Command):
    """unpause specified container"""
    log = logging.getLogger(__name__ + '.UnpauseContainer')

    def get_parser(self, prog_name):
        parser = super(UnpauseContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', nargs='+', help='ID or name of the (container)s to unpause.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        containers = parsed_args.container
        for container in containers:
            try:
                client.containers.unpause(container)
                print(_('Request to unpause container %s has been accepted') % container)
            except Exception as e:
                print('unpause for container %(container)s failed: %(e)s' % {'container': container, 'e': e})