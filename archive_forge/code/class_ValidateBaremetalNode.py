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
class ValidateBaremetalNode(command.Lister):
    """Validate a node's driver interfaces"""
    log = logging.getLogger(__name__ + '.ValidateBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ValidateBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        interfaces = baremetal_client.node.validate(parsed_args.node)._info
        data = []
        for key, value in interfaces.items():
            interface = {'interface': key}
            interface.update(value)
            data.append(interface)
        field_labels = ['Interface', 'Result', 'Reason']
        fields = ['interface', 'result', 'reason']
        data = oscutils.sort_items(data, 'interface')
        return (field_labels, (oscutils.get_dict_properties(s, fields) for s in data))