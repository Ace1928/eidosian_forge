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
def _find_share_type(cs, stype):
    """Get a share type by name or ID."""
    return apiclient_utils.find_resource(cs.share_types, stype)