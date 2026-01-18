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
class PassthruListBaremetalNode(command.Lister):
    """List vendor passthru methods for a node"""
    log = logging.getLogger(__name__ + '.PassthruListBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(PassthruListBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('node', metavar='<node>', help=_('Name or UUID of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        methods = baremetal_client.node.get_vendor_passthru_methods(parsed_args.node)
        data = []
        for method, response in methods.items():
            response['name'] = method
            response['http_methods'] = oscutils.format_list(response['http_methods'])
            data.append(response)
        return (res_fields.VENDOR_PASSTHRU_METHOD_RESOURCE.labels, (oscutils.get_dict_properties(s, res_fields.VENDOR_PASSTHRU_METHOD_RESOURCE.fields) for s in data))