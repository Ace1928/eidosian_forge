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
class RescueBaremetalNode(ProvisionStateWithWait):
    """Set provision state of baremetal node to 'rescue'"""
    log = logging.getLogger(__name__ + '.RescueBaremetalNode')
    PROVISION_STATE = 'rescue'

    def get_parser(self, prog_name):
        parser = super(RescueBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--rescue-password', metavar='<rescue-password>', required=True, default=None, help='The password that will be used to login to the rescue ramdisk. The value should be a non-empty string.')
        return parser