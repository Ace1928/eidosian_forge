from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _is_v13_and_above(self):
    version = tmos_version(self.client)
    if Version(version) >= Version('13.0.0'):
        return True
    return False