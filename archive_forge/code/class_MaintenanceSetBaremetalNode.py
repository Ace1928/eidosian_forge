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
class MaintenanceSetBaremetalNode(command.Command):
    """Set baremetal node to maintenance mode"""
    log = logging.getLogger(__name__ + '.MaintenanceSetBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(MaintenanceSetBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes."))
        parser.add_argument('--reason', metavar='<reason>', default=None, help=_('Reason for setting maintenance mode.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        for node in parsed_args.nodes:
            baremetal_client.node.set_maintenance(node, True, maint_reason=parsed_args.reason)