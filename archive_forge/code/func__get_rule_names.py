from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_rule_names(self, rules):
    if 'items' in rules:
        rules['items'].sort(key=lambda x: x['ordinal'])
        result = [x['name'] for x in rules['items']]
        return result
    else:
        return []