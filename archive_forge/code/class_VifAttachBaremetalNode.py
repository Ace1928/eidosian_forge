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
class VifAttachBaremetalNode(command.Command):
    """Attach VIF to a given node"""
    log = logging.getLogger(__name__ + '.VifAttachBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(VifAttachBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        parser.add_argument('vif_id', metavar='<vif-id>', help=_('Name or UUID of the VIF to attach to a node.'))
        parser.add_argument('--port-uuid', metavar='<port-uuid>', help=_('UUID of the baremetal port to attach the VIF to.'))
        parser.add_argument('--vif-info', metavar='<key=value>', action='append', help=_("Record arbitrary key/value metadata. Can be specified multiple times. The mandatory 'id' parameter cannot be specified as a key."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = utils.key_value_pairs_to_dict(parsed_args.vif_info or [])
        if parsed_args.port_uuid:
            fields['port_uuid'] = parsed_args.port_uuid
        baremetal_client.node.vif_attach(parsed_args.node, parsed_args.vif_id, **fields)