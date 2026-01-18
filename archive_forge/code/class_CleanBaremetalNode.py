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
class CleanBaremetalNode(ProvisionStateWithWait):
    """Set provision state of baremetal node to 'clean'"""
    log = logging.getLogger(__name__ + '.CleanBaremetalNode')
    PROVISION_STATE = 'clean'

    def get_parser(self, prog_name):
        parser = super(CleanBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--clean-steps', metavar='<clean-steps>', required=True, default=None, help=_("The clean steps. May be the path to a YAML file containing the clean steps; OR '-', with the clean steps being read from standard input; OR a JSON string. The value should be a list of clean-step dictionaries; each dictionary should have keys 'interface' and 'step', and optional key 'args'."))
        return parser