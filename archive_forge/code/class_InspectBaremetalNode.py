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
class InspectBaremetalNode(ProvisionStateWithWait):
    """Set provision state of baremetal node to 'inspect'"""
    log = logging.getLogger(__name__ + '.InspectBaremetalNode')
    PROVISION_STATE = 'inspect'