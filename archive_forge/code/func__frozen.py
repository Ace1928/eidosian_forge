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
def _frozen(self, count=0):
    """Ensure a container is frozen.

        If the container does not exist the container will be created.

        :param count: number of times this command has been called by itself.
        :type count: ``int``
        """
    self.check_count(count=count, method='frozen')
    if self._container_exists(container_name=self.container_name, lxc_path=self.lxc_path):
        self._execute_command()
        self._config()
        container_state = self._get_state()
        if container_state == 'frozen':
            pass
        elif container_state == 'running':
            self.container.freeze()
            self.state_change = True
        else:
            self._container_startup()
            self.container.freeze()
            self.state_change = True
        self._check_archive()
        self._check_clone()
    else:
        self._create()
        count += 1
        self._frozen(count)