from __future__ import absolute_import, division, print_function
import time
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
Checks to see if guest is disabled

        A disabled guest is fully disabled once their Stats go offline.
        Until that point they are still in the process of disabling.

        :return:
        