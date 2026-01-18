from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _create_time_window(window):
    period = window[-1].lower()
    multiple = int(window[0:-1])
    if period == 'h':
        return HOUR * multiple
    if period == 'd':
        return DAY * multiple
    if period == 'w':
        return WEEK * multiple
    if period == 'y':
        return YEAR * multiple