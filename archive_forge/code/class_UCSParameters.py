from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class UCSParameters(BaseParameters):
    api_map = {'filename': 'file_name', 'encrypted': 'encrypted', 'file_size': 'file_size', 'apiRawValues': 'variables'}
    returnables = ['file_name', 'encrypted', 'file_size', 'file_created_date']

    @property
    def file_name(self):
        name = self._values['variables']['filename'].split('/')[-1]
        return name

    @property
    def encrypted(self):
        return self._values['variables']['encrypted']

    @property
    def file_size(self):
        val = self._values['variables']['file_size']
        size = re.findall('\\d+', val)[0]
        return size

    @property
    def file_created_date(self):
        date = self._values['variables']['file_created_date']
        return date