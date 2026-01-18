from __future__ import absolute_import, division, print_function
import re
import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _compare_date_time(self, value1, value2, time=False):
    if time:
        p1 = '(\\d{2}):(\\d{2})'
        m1 = re.match(p1, value1)
        m2 = re.match(p1, value2)
        if m1 and m2:
            start = tuple((int(i) for i in m1.group(1, 2)))
            end = tuple((int(i) for i in m2.group(1, 2)))
            if datetime.time(*start) > datetime.time(*end):
                raise F5ModuleError('End time must be later than start time.')
    else:
        p1 = '(\\d{4})-(\\d{1,2})-(\\d{1,2})[:, T](\\d{2}):(\\d{2}):(\\d{2})'
        m1 = re.match(p1, value1)
        m2 = re.match(p1, value2)
        if m1 and m2:
            start = tuple((int(i) for i in m1.group(1, 2, 3, 4, 5, 6)))
            end = tuple((int(i) for i in m2.group(1, 2, 3, 4, 5, 6)))
            if datetime.datetime(*start) > datetime.datetime(*end):
                raise F5ModuleError('End date must be later than start date.')