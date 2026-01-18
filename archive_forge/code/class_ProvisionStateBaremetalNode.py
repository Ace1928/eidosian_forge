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
class ProvisionStateBaremetalNode(command.Command):
    """Base provision state class"""
    log = logging.getLogger(__name__ + '.ProvisionStateBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(ProvisionStateBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes."))
        parser.add_argument('--provision-state', default=self.PROVISION_STATE, required=False, choices=[self.PROVISION_STATE], help=argparse.SUPPRESS)
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        clean_steps = getattr(parsed_args, 'clean_steps', None)
        clean_steps = utils.handle_json_arg(clean_steps, 'clean steps')
        deploy_steps = getattr(parsed_args, 'deploy_steps', None)
        deploy_steps = utils.handle_json_arg(deploy_steps, 'deploy steps')
        config_drive = getattr(parsed_args, 'config_drive', None)
        if config_drive:
            try:
                config_drive_dict = json.loads(config_drive)
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(config_drive_dict, dict):
                    config_drive = config_drive_dict
        rescue_password = getattr(parsed_args, 'rescue_password', None)
        for node in parsed_args.nodes:
            baremetal_client.node.set_provision_state(node, parsed_args.provision_state, configdrive=config_drive, cleansteps=clean_steps, deploysteps=deploy_steps, rescue_password=rescue_password)