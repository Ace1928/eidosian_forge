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
def _find_share_snapshot_instance(cs, snapshot_instance):
    """Get a share snapshot instance by ID."""
    return apiclient_utils.find_resource(cs.share_snapshot_instances, snapshot_instance)