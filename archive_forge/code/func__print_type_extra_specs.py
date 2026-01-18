from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def _print_type_extra_specs(share_type):
    """Prints share type extra specs or share group type specs."""
    try:
        return _print_dict(share_type.get_keys())
    except exceptions.NotFound:
        return None