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
@staticmethod
def _container_exists(container_name, lxc_path=None):
    """Check if a container exists.

        :param container_name: Name of the container.
        :type: ``str``
        :returns: True or False if the container is found.
        :rtype: ``bol``
        """
    return any((c == container_name for c in lxc.list_containers(config_path=lxc_path)))