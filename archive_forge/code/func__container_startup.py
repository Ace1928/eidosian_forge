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
def _container_startup(self, timeout=60):
    """Ensure a container is started.

        :param timeout: Time before the destroy operation is abandoned.
        :type timeout: ``int``
        """
    self.container = self.get_container_bind()
    for dummy in range(timeout):
        if self._get_state() == 'running':
            return True
        self.container.start()
        self.state_change = True
        time.sleep(1)
    self.failure(lxc_container=self._container_data(), error='Failed to start container [ %s ]' % self.container_name, rc=1, msg='The container [ %s ] failed to start. Check to lxc is available and that the container is in a functional state.' % self.container_name)