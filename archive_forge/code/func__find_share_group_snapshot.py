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
@api_versions.wraps('2.31')
def _find_share_group_snapshot(cs, share_group_snapshot):
    """Get a share group snapshot by name or ID."""
    return apiclient_utils.find_resource(cs.share_group_snapshots, share_group_snapshot)