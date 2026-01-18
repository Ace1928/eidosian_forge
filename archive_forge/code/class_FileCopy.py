from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
class FileCopy:

    def __init__(self, module):
        self._module = module
        self._connection = get_resource_connection(self._module)
        device_info = self._connection.get_device_info()
        self._model = device_info.get('network_os_model', '')
        self._platform = device_info.get('network_os_platform', '')