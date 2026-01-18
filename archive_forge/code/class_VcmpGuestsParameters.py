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
class VcmpGuestsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'allowedSlots': 'allowed_slots', 'assignedSlots': 'assigned_slots', 'bootPriority': 'boot_priority', 'coresPerSlot': 'cores_per_slot', 'initialImage': 'initial_image', 'initialHotfix': 'hotfix_image', 'managementGw': 'mgmt_route', 'managementIp': 'mgmt_address', 'managementNetwork': 'mgmt_network', 'minSlots': 'min_number_of_slots', 'slots': 'number_of_slots', 'sslMode': 'ssl_mode', 'virtualDisk': 'virtual_disk'}
    returnables = ['name', 'full_path', 'allowed_slots', 'assigned_slots', 'boot_priority', 'cores_per_slot', 'hostname', 'hotfix_image', 'initial_image', 'mgmt_route', 'mgmt_address', 'mgmt_network', 'vlans', 'min_number_of_slots', 'number_of_slots', 'ssl_mode', 'state', 'virtual_disk']