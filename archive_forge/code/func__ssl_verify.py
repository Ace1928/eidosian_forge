from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def _ssl_verify(self, option):
    try:
        verify = to_bool(option)
    except TypeError:
        verify = option
    if verify is False:
        requests.packages.urllib3.disable_warnings()
        self._display.warning(u'SSL verification of %s disabled' % self.foreman_url)
    return verify