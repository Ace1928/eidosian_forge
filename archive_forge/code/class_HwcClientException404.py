from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
class HwcClientException404(HwcClientException):

    def __init__(self, message):
        super(HwcClientException404, self).__init__(404, message)

    def __str__(self):
        return '[HwcClientException404] message=%s' % self._message