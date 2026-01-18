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
def _set_default_profiles(self):
    if self.want.type == 'standard':
        if not self.want.profiles:
            if self.want.ip_protocol == 6:
                self.want.update({'profiles': ['tcp']})
            if self.want.ip_protocol == 17:
                self.want.update({'profiles': ['udp']})
            if self.want.ip_protocol == 132:
                self.want.update({'profiles': ['sctp']})