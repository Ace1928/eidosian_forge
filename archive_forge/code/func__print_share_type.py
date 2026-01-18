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
def _print_share_type(stype, default_share_type=None, show_des=False):

    def _is_default(share_type):
        if hasattr(share_type, 'is_default'):
            return 'YES' if share_type.is_default else '-'
        return '-'
    stype_dict = {'ID': stype.id, 'Name': stype.name, 'Visibility': _is_share_type_public(stype), 'is_default': _is_default(stype), 'required_extra_specs': _print_type_required_extra_specs(stype), 'optional_extra_specs': _print_type_optional_extra_specs(stype)}
    if show_des:
        stype_dict['Description'] = stype.description
    cliutils.print_dict(stype_dict)