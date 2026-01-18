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
def _started(self, count=0):
    """Ensure a container is started.

        If the container does not exist the container will be created.

        :param count: number of times this command has been called by itself.
        :type count: ``int``
        """
    self.check_count(count=count, method='start')
    if self._container_exists(container_name=self.container_name, lxc_path=self.lxc_path):
        container_state = self._get_state()
        if container_state == 'running':
            pass
        elif container_state == 'frozen':
            self._unfreeze()
        elif not self._container_startup():
            self.failure(lxc_container=self._container_data(), error='Failed to start container [ %s ]' % self.container_name, rc=1, msg='The container [ %s ] failed to start. Check to lxc is available and that the container is in a functional state.' % self.container_name)
        self._execute_command()
        self._config()
        self._check_archive()
        self._check_clone()
    else:
        self._create()
        count += 1
        self._started(count)