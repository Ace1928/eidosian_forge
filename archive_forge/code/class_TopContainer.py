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
class TopContainer(command.Command):
    """Display the running processes inside the container"""
    log = logging.getLogger(__name__ + '.TopContainer')

    def get_parser(self, prog_name):
        parser = super(TopContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to display processes.')
        parser.add_argument('--pid', metavar='<pid>', action='append', default=[], help='The args of the ps id.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        if parsed_args.pid:
            output = client.containers.top(parsed_args.container, ' '.join(parsed_args.pid))
        else:
            output = client.containers.top(parsed_args.container)
        for titles in output['Titles']:
            (print('%-20s') % titles,)
        if output['Processes']:
            for process in output['Processes']:
                print('')
                for info in process:
                    (print('%-20s') % info,)
        else:
            print('')