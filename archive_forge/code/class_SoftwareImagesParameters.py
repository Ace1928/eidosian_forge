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
class SoftwareImagesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'buildDate': 'build_date', 'fileSize': 'file_size', 'lastModified': 'last_modified'}
    returnables = ['name', 'full_path', 'build', 'build_date', 'checksum', 'file_size', 'last_modified', 'product', 'verified', 'version']

    @property
    def file_size(self):
        if self._values['file_size'] is None:
            return None
        matches = re.match('\\d+', self._values['file_size'])
        if matches:
            return int(matches.group(0))

    @property
    def build_date(self):
        """Normalizes the build_date string

        The ISOs usually ship with a broken format

        ex: Tue May 15 15 26 30 PDT 2018

        This will re-format that time so that it looks like ISO 8601 without
        microseconds

        ex: 2018-05-15T15:26:30

        :return:
        """
        if self._values['build_date'] is None:
            return None
        d = self._values['build_date'].split(' ')
        d.pop(6)
        result = datetime.datetime.strptime(' '.join(d), '%a %b %d %H %M %S %Y').isoformat()
        return result

    @property
    def last_modified(self):
        """Normalizes the last_modified string

        The strings that the system reports look like the following

        ex: Tue May 15 15:26:30 2018

        This property normalizes this value to be isoformat

        ex: 2018-05-15T15:26:30

        :return:
        """
        if self._values['last_modified'] is None:
            return None
        result = datetime.datetime.strptime(self._values['last_modified'], '%a %b %d %H:%M:%S %Y').isoformat()
        return result