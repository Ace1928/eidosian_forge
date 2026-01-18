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
class RemoveTraitBaremetalNode(command.Command):
    """Remove trait(s) from a node."""
    log = logging.getLogger(__name__ + '.RemoveTraitBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(RemoveTraitBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        all_or_trait = parser.add_mutually_exclusive_group(required=True)
        all_or_trait.add_argument('--all', dest='remove_all', action='store_true', help=_('Remove all traits'))
        all_or_trait.add_argument('traits', metavar='<trait>', nargs='*', default=[], help=_('Trait(s) to remove'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        if parsed_args.remove_all:
            baremetal_client.node.remove_all_traits(parsed_args.node)
        else:
            for trait in parsed_args.traits:
                try:
                    baremetal_client.node.remove_trait(parsed_args.node, trait)
                    print(_('Removed trait %s') % trait)
                except exc.ClientException as e:
                    failures.append(_('Failed to remove trait %(trait)s: %(error)s') % {'trait': trait, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))