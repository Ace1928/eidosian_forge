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
def _unfreeze(self):
    """Unfreeze a container.

        :returns: True or False based on if the container was unfrozen.
        :rtype: ``bol``
        """
    unfreeze = self.container.unfreeze()
    if unfreeze:
        self.state_change = True
    return unfreeze