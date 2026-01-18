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
def _add_interface_args(self, parser, iface, set_help, reset_help):
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--%s-interface' % iface, metavar='<%s_interface>' % iface, help=set_help)
    grp.add_argument('--reset-%s-interface' % iface, action='store_true', help=reset_help)