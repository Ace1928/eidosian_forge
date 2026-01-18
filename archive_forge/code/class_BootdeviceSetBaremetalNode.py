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
class BootdeviceSetBaremetalNode(command.Command):
    """Set the boot device for a node"""
    log = logging.getLogger(__name__ + '.BootdeviceSetBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(BootdeviceSetBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes"))
        parser.add_argument('device', metavar='<device>', choices=v1_utils.BOOT_DEVICES, help=_('One of %s') % oscutils.format_list(v1_utils.BOOT_DEVICES))
        parser.add_argument('--persistent', dest='persistent', action='store_true', default=False, help=_('Make changes persistent for all future boots'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        for node in parsed_args.nodes:
            baremetal_client.node.set_boot_device(node, parsed_args.device, parsed_args.persistent)