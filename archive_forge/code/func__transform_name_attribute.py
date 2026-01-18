from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
def _transform_name_attribute(self, entry):
    if isinstance(entry, dict):
        tmp = copy.deepcopy(entry)
        for k, v in iteritems(tmp):
            if k == 'tmName':
                entry['name'] = entry.pop('tmName')
            self._transform_name_attribute(v)
    elif isinstance(entry, list):
        for k in entry:
            self._transform_name_attribute(k)
    else:
        return