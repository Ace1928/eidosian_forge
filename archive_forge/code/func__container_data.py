from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _container_data(self):
    """Returns a dict of container information.

        :returns: container data
        :rtype: ``dict``
        """
    return {'interfaces': self.container.get_interfaces(), 'ips': self.container.get_ips(), 'state': self._get_state(), 'init_pid': int(self.container.init_pid), 'name': self.container_name}