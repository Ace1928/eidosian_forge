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
class ListFirmwareComponentBaremetalNode(command.Lister):
    """List all Firmware Components of a node"""
    log = logging.getLogger(__name__ + '.ListFirmwareComponentBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ListFirmwareComponentBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        labels = res_fields.FIRMWARE_RESOURCE.labels
        fields = res_fields.FIRMWARE_RESOURCE.fields
        baremetal_client = self.app.client_manager.baremetal
        components = baremetal_client.node.list_firmware_components(parsed_args.node)
        return (labels, (oscutils.get_dict_properties(s, fields) for s in components))