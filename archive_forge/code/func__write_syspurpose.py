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
def _write_syspurpose(self, new_syspurpose):
    """
        This function tries to update current new_syspurpose attributes to
        json file.
        """
    with open(self.path, 'w') as fp:
        fp.write(json.dumps(new_syspurpose, indent=2, ensure_ascii=False, sort_keys=True))