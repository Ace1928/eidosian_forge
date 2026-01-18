from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
@staticmethod
def _transform_state(server):
    if 'status' in server:
        server['state'] = server['status']
        del server['status']
    else:
        server['state'] = 'absent'
    return server