from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
Waits specifically for MGMT

        Modifying memory reserve for mgmt can take longer to actually start up than all the previous checks take.
        This check here is specifically waiting for a MGMT API to stop raising
        errors.
        :return:
        