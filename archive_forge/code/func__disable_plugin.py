from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def _disable_plugin(self, msg):
    self.disabled = True
    if msg:
        self._display.warning(msg + u' Disabling the Foreman callback plugin.')
    else:
        self._display.warning(u'Disabling the Foreman callback plugin.')