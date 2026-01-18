from __future__ import absolute_import, division, print_function
import os
import re
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _is_config_reloading_success_on_device(self, output):
    succeed = 'Last Configuration Load Status\\s+full-config-load-succeed'
    matches = re.search(succeed, output)
    if matches:
        return True
    return False