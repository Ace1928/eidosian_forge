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
def _destroyed(self, timeout=60):
    """Ensure a container is destroyed.

        :param timeout: Time before the destroy operation is abandoned.
        :type timeout: ``int``
        """
    for dummy in range(timeout):
        if not self._container_exists(container_name=self.container_name, lxc_path=self.lxc_path):
            break
        self._check_archive()
        self._check_clone()
        if self._get_state() != 'stopped':
            self.state_change = True
            self.container.stop()
        if self.container.destroy():
            self.state_change = True
        time.sleep(1)
    else:
        self.failure(lxc_container=self._container_data(), error='Failed to destroy container [ %s ]' % self.container_name, rc=1, msg='The container [ %s ] failed to be destroyed. Check that lxc is available and that the container is in a functional state.' % self.container_name)