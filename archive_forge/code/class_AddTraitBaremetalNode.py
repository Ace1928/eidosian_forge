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
class AddTraitBaremetalNode(command.Command):
    """Add traits to a node."""
    log = logging.getLogger(__name__ + '.AddTraitBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(AddTraitBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        parser.add_argument('traits', nargs='+', metavar='<trait>', help=_('Trait(s) to add'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for trait in parsed_args.traits:
            try:
                baremetal_client.node.add_trait(parsed_args.node, trait)
                print(_('Added trait %s') % trait)
            except exc.ClientException as e:
                failures.append(_('Failed to add trait %(trait)s: %(error)s') % {'trait': trait, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))