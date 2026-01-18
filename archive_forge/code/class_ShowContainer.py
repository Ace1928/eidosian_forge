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
class ShowContainer(command.ShowOne):
    """Show a container"""
    log = logging.getLogger(__name__ + '.ShowContainer')

    def get_parser(self, prog_name):
        parser = super(ShowContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to show.')
        parser.add_argument('--all-projects', action='store_true', default=False, help='Show container(s) in all projects by name.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.container
        opts['all_projects'] = parsed_args.all_projects
        opts = zun_utils.remove_null_parms(**opts)
        container = client.containers.get(**opts)
        zun_utils.format_container_addresses(container)
        columns = _container_columns(container)
        return (columns, utils.get_item_properties(container, columns))