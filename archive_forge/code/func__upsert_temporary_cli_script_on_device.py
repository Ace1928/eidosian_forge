from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _upsert_temporary_cli_script_on_device(self):
    args = {'name': '__ansible_mkqkview', 'apiAnonymous': '\n                proc script::run {} {\n                    set cmd [lreplace $tmsh::argv 0 0]; eval "exec $cmd 2> /dev/null"\n                }\n            '}
    result = self._create_temporary_cli_script_on_device(args)
    if result:
        return True
    return self._update_temporary_cli_script_on_device(args)