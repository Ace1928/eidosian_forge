from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
class AnsibleModuleError(Exception):

    def __init__(self, results):
        self.results = results

    def __repr__(self):
        return 'AnsibleModuleError(results={0})'.format(self.results)