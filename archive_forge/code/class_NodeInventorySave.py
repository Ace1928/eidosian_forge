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
class NodeInventorySave(command.Command):
    """Get hardware inventory of a node (in JSON format) or save it to file."""
    log = logging.getLogger(__name__ + '.NodeInventorySave')

    def get_parser(self, prog_name):
        parser = super(NodeInventorySave, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        parser.add_argument('--file', metavar='<filename>', help='Save inspection data to file with name (default: stdout).')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        inventory = baremetal_client.node.get_inventory(parsed_args.node)
        if parsed_args.file:
            with open(parsed_args.file, 'w') as fp:
                json.dump(inventory, fp)
        else:
            json.dump(inventory, sys.stdout)