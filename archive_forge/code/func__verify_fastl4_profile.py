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
def _verify_fastl4_profile(self):
    if self.want.type != 'performance-l4':
        return
    if self.want.profiles is None:
        return
    have = set(self.read_fastl4_profiles_from_device())
    want = set([x['fullPath'] for x in self.want.profiles])
    if have.intersection(want):
        return True
    raise F5ModuleError("A performance-l4 profile, such as 'fastL4', must be specified when 'type' is 'performance-l4'.")