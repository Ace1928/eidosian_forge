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
class NodeChildrenList(command.ShowOne):
    """Get a list of nodes associated as children."""
    log = logging.getLogger(__name__ + '.NodeChildrenList')

    def get_parser(self, prog_name):
        parser = super(NodeChildrenList, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        labels = res_fields.CHILDREN_RESOURCE.labels
        data = baremetal_client.node.list_children_of_node(parsed_args.node)
        return (labels, [[node] for node in data])