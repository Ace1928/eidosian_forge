from __future__ import absolute_import, division, print_function
from datetime import datetime
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _detect_errors(self, stdout):
    errors = ['Unexpected Error:']
    msg = [x for x in stdout for y in errors if y in x]
    if msg:
        raise F5ModuleError(' '.join(msg))