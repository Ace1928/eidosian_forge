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
def _find_share_group_type(cs, sg_type):
    """Get a share group type by name or ID."""
    return apiclient_utils.find_resource(cs.share_group_types, sg_type)