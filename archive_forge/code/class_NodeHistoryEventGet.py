import argparse
import itertools
import json
import logging
import sys
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class NodeHistoryEventGet(command.ShowOne):
    """Get history event for a baremetal node."""
    log = logging.getLogger(__name__ + '.NodeHistoryEventGet')

    def get_parser(self, prog_name):
        parser = super(NodeHistoryEventGet, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node.'))
        parser.add_argument('event', metavar='<event>', help=_('UUID of the event.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        data = baremetal_client.node.get_history_event(parsed_args.node, parsed_args.event)
        data.pop('links')
        return self.dict2columns(data)