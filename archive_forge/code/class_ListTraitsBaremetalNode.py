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
class ListTraitsBaremetalNode(command.Lister):
    """List a node's traits."""
    log = logging.getLogger(__name__ + '.ListTraitsBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ListTraitsBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        labels = res_fields.TRAIT_RESOURCE.labels
        baremetal_client = self.app.client_manager.baremetal
        traits = baremetal_client.node.get_traits(parsed_args.node)
        return (labels, [[trait] for trait in traits])