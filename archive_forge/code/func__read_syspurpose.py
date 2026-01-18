from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def _read_syspurpose(self):
    """
        Read current syspurpuse from json file.
        """
    current_syspurpose = {}
    try:
        with open(self.path, 'r') as fp:
            content = fp.read()
    except IOError:
        pass
    else:
        current_syspurpose = json.loads(content)
    return current_syspurpose