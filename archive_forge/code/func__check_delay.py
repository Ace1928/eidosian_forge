from __future__ import (absolute_import, division, print_function)
import random
import time
from datetime import datetime, timedelta, timezone
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_list, check_type_str
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _check_delay(self, key, default):
    """Ensure that the value is positive or zero"""
    value = int(self._task.args.get(key, self._task.args.get(key + '_sec', default)))
    if value < 0:
        value = 0
    return value