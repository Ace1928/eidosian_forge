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
class SoftwareVolumesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'basebuild': 'base_build'}
    returnables = ['full_path', 'name', 'active', 'base_build', 'build', 'product', 'status', 'version', 'install_volume', 'default_boot_location']

    @property
    def install_volume(self):
        if self._values['media'] is None:
            return None
        return self._values['media'].get('name', None)

    @property
    def default_boot_location(self):
        if self._values['media'] is None:
            return None
        return flatten_boolean(self._values['media'].get('defaultBootLocation', None))

    @property
    def active(self):
        if self._values['active'] is True:
            return 'yes'
        return 'no'