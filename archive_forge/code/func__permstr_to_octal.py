from __future__ import absolute_import, division, print_function
import binascii
import codecs
import datetime
import fnmatch
import grp
import os
import platform
import pwd
import re
import stat
import time
import traceback
from functools import partial
from zipfile import ZipFile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_file
def _permstr_to_octal(self, modestr, umask):
    """ Convert a Unix permission string (rw-r--r--) into a mode (0644) """
    revstr = modestr[::-1]
    mode = 0
    for j in range(0, 3):
        for i in range(0, 3):
            if revstr[i + 3 * j] in ['r', 'w', 'x', 's', 't']:
                mode += 2 ** (i + 3 * j)
    return mode & ~umask