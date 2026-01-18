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
def _is_default(share_group_type):
    if hasattr(share_group_type, 'is_default'):
        return 'YES' if share_group_type.is_default else '-'
    return '-'