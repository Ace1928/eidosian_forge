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
def _print_share_type_list(stypes, default_share_type=None, columns=None, description=False):

    def _is_default(share_type):
        if hasattr(share_type, 'is_default'):
            return 'YES' if share_type.is_default else '-'
        elif default_share_type:
            default = default_share_type.id
            return 'YES' if share_type.id == default else '-'
        else:
            return '-'
    formatters = {'visibility': _is_share_type_public, 'is_default': _is_default, 'required_extra_specs': _print_type_required_extra_specs, 'optional_extra_specs': _print_type_optional_extra_specs}
    for stype in stypes:
        stype = stype.to_dict()
        stype['visibility'] = stype.pop('is_public', 'unknown')
    fields = ['ID', 'Name', 'visibility', 'is_default', 'required_extra_specs', 'optional_extra_specs']
    if description:
        fields.append('Description')
    if columns is not None:
        fields = _split_columns(columns=columns, title=False)
    cliutils.print_list(stypes, fields, formatters)