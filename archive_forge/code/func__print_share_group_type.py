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
def _print_share_group_type(share_group_type, default_share_type=None):

    def _is_default(share_group_type):
        if hasattr(share_group_type, 'is_default'):
            return 'YES' if share_group_type.is_default else '-'
        return '-'
    share_group_type_dict = {'ID': share_group_type.id, 'Name': share_group_type.name, 'Visibility': _is_share_type_public(share_group_type), 'is_default': _is_default(share_group_type)}
    cliutils.print_dict(share_group_type_dict)