from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _check_clone_pool_contexts(self):
    client = 0
    server = 0
    for item in self._values['clone_pools']:
        if item['context'] == 'clientside':
            client += 1
        if item['context'] == 'serverside':
            server += 1
    if client > 1 or server > 1:
        raise F5ModuleError('You must specify only one clone pool for each context.')