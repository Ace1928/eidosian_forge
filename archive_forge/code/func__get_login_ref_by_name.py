from __future__ import absolute_import, division, print_function
import os
from .common import (
from .constants import (
from .icontrol import iControlRestSession
@staticmethod
def _get_login_ref_by_name(info, provider):
    for x in info['providers']:
        if x['name'] == provider:
            return x['link']
    return None