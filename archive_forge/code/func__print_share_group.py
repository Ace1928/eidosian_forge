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
def _print_share_group(cs, share_group):
    info = share_group._info.copy()
    info.pop('links', None)
    if info.get('share_types'):
        info['share_types'] = '\n'.join(info['share_types'])
    cliutils.print_dict(info)