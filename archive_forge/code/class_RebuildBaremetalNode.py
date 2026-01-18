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
class RebuildBaremetalNode(ProvisionStateWithWait):
    """Set provision state of baremetal node to 'rebuild'"""
    log = logging.getLogger(__name__ + '.RebuildBaremetalNode')
    PROVISION_STATE = 'rebuild'

    def get_parser(self, prog_name):
        parser = super(RebuildBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--config-drive', metavar='<config-drive>', default=None, help=CONFIG_DRIVE_ARG_HELP)
        parser.add_argument('--deploy-steps', metavar='<deploy-steps>', required=False, default=None, help=_("The deploy steps in JSON format. May be the path to a file containing the deploy steps; OR '-', with the deploy steps being read from standard input; OR a string. The value should be a list of deploy-step dictionaries; each dictionary should have keys 'interface', 'step', 'priority' and optional key 'args'."))
        return parser