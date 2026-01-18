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
class LogsContainer(command.Command):
    """Get logs of a container"""
    log = logging.getLogger(__name__ + '.LogsContainer')

    def get_parser(self, prog_name):
        parser = super(LogsContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to get logs for.')
        parser.add_argument('--stdout', action='store_true', help='Only stdout logs of container.')
        parser.add_argument('--stderr', action='store_true', help='Only stderr logs of container.')
        parser.add_argument('--since', metavar='<since>', default=None, help='Show logs since a given datetime or integer epoch (in seconds).')
        parser.add_argument('--timestamps', dest='timestamps', action='store_true', default=False, help='Show timestamps.')
        parser.add_argument('--tail', metavar='<tail>', default='all', help='Number of lines to show from the end of the logs.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.container
        opts['stdout'] = parsed_args.stdout
        opts['stderr'] = parsed_args.stderr
        opts['since'] = parsed_args.since
        opts['timestamps'] = parsed_args.timestamps
        opts['tail'] = parsed_args.tail
        opts = zun_utils.remove_null_parms(**opts)
        logs = client.containers.logs(**opts)
        print(logs)