from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
class HwcClientException(Exception):

    def __init__(self, code, message):
        super(HwcClientException, self).__init__()
        self._code = code
        self._message = message

    def __str__(self):
        msg = ' code=%s,' % str(self._code) if self._code != 0 else ''
        return '[HwcClientException]%s message=%s' % (msg, self._message)