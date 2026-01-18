from __future__ import absolute_import, division, print_function
import copy
import os
import ssl
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.connection import exec_command
from ..module_utils.common import (
def _parse_cert_date(self, to_parse):
    c_time = to_parse.split('\n')[1].split('=')[1]
    result = ssl.cert_time_to_seconds(c_time)
    return result