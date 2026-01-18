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
def _print_share_group_type_list(share_group_types, default_share_group_type=None, columns=None):

    def _is_default(share_group_type):
        if hasattr(share_group_type, 'is_default'):
            return 'YES' if share_group_type.is_default else '-'
        elif default_share_group_type:
            default = default_share_group_type.id
            return 'YES' if share_group_type.id == default else '-'
        else:
            return '-'
    formatters = {'visibility': _is_share_type_public, 'is_default': _is_default}
    for sg_type in share_group_types:
        sg_type = sg_type.to_dict()
        sg_type['visibility'] = sg_type.pop('is_public', 'unknown')
    fields = ['ID', 'Name', 'visibility', 'is_default']
    if columns is not None:
        fields = _split_columns(columns=columns, title=False)
    cliutils.print_list(share_group_types, fields, formatters, sortby_index=None)