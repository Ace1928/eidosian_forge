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
def _print_type_show(stype, default_share_type=None):
    if hasattr(stype, 'is_default'):
        is_default = 'YES' if stype.is_default else 'NO'
    elif default_share_type:
        is_default = 'YES' if stype.id == default_share_type.id else 'NO'
    else:
        is_default = 'NO'
    stype_dict = {'id': stype.id, 'name': stype.name, 'visibility': _is_share_type_public(stype), 'is_default': is_default, 'description': stype.description, 'required_extra_specs': _print_type_required_extra_specs(stype), 'optional_extra_specs': _print_type_optional_extra_specs(stype)}
    cliutils.print_dict(stype_dict)