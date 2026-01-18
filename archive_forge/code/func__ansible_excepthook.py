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
def _ansible_excepthook(exc_type, exc_value, tb):
    if issubclass(exc_type, AnsibleModuleError):
        module.fail_json(**exc_value.results)
    else:
        sys.__excepthook__(exc_type, exc_value, tb)