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
class ActionShow(command.ShowOne):
    """Show a action"""
    log = logging.getLogger(__name__ + '.ShowAction')

    def get_parser(self, prog_name):
        parser = super(ActionShow, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to show.')
        parser.add_argument('request_id', metavar='<request_id>', help='request ID of action to describe.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        container = parsed_args.container
        request_id = parsed_args.request_id
        action = client.actions.get(container, request_id)
        columns = _action_columns(action)
        return (columns, utils.get_item_properties(action, columns))