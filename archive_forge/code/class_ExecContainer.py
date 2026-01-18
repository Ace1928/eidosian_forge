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
class ExecContainer(command.Command):
    """Execute command in a running container"""
    log = logging.getLogger(__name__ + '.ExecContainer')

    def get_parser(self, prog_name):
        parser = super(ExecContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to execute command in.')
        parser.add_argument('command', metavar='<command>', nargs=argparse.REMAINDER, help='The command to execute.')
        parser.add_argument('--interactive', dest='interactive', action='store_true', default=False, help='Keep STDIN open and allocate a pseudo-TTY for interactive')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        container = parsed_args.container
        opts = {}
        opts['command'] = zun_utils.parse_command(parsed_args.command)
        if parsed_args.interactive:
            opts['interactive'] = True
            opts['run'] = False
        response = client.containers.execute(container, **opts)
        if parsed_args.interactive:
            exec_id = response['exec_id']
            url = response['proxy_url']
            websocketclient.do_exec(client, url, container, exec_id, '~', 0.5)
        else:
            output = response['output']
            exit_code = response['exit_code']
            print(output)
            return exit_code