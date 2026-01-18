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
class PowerOffBaremetalNode(PowerBaremetalNode):
    """Power off a node"""
    log = logging.getLogger(__name__ + '.PowerOffBaremetalNode')
    POWER_STATE = 'off'

    def get_parser(self, prog_name):
        parser = super(PowerOffBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--soft', dest='soft', action='store_true', default=False, help=_('Request graceful power-off.'))
        return parser