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
class ConsoleShowBaremetalNode(command.ShowOne):
    """Show console information for a node"""
    log = logging.getLogger(__name__ + '.ConsoleShowBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ConsoleShowBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        info = baremetal_client.node.get_console(parsed_args.node)
        return zip(*sorted(info.items()))