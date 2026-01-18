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
@api_versions.wraps('2.9')
def _print_share_instance(cs, instance):
    info = instance._info.copy()
    info.pop('links', None)
    if info.get('export_locations'):
        info['export_locations'] = cliutils.convert_dict_list_to_string(info['export_locations'], ignored_keys=['replica_state', 'availability_zone', 'share_replica_id'])
    cliutils.print_dict(info)