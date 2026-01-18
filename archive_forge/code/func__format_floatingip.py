import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
def _format_floatingip(fip):
    fip.pop('links', None)