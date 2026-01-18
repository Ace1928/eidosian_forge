from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def __default(self, param):
    attr1 = getattr(self.want, param)
    try:
        attr2 = getattr(self.have, param)
        if attr1 != attr2:
            return attr1
    except AttributeError:
        return attr1