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
def _roundup(num):
    """Return a rounded floating point number.

        :param num: Number to round up.
        :type: ``float``
        :returns: Rounded up number.
        :rtype: ``int``
        """
    num, part = str(num).split('.')
    num = int(num)
    if int(part) != 0:
        num += 1
    return num