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
class DeleteBaremetalNode(command.Command):
    """Unregister baremetal node(s)"""
    log = logging.getLogger(__name__ + '.DeleteBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_('Node(s) to delete (name or UUID)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for node in parsed_args.nodes:
            try:
                baremetal_client.node.delete(node)
                print(_('Deleted node %s') % node)
            except exc.ClientException as e:
                failures.append(_('Failed to delete node %(node)s: %(error)s') % {'node': node, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))