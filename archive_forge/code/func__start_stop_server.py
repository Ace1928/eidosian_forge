from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def _start_stop_server(self, server_info, target_state='running', ignore_diff=False):
    actions = {'stopped': 'stop', 'running': 'start'}
    server_state = server_info.get('state')
    if server_state != target_state:
        self._result['changed'] = True
        if not ignore_diff:
            self._result['diff']['before'].update({'state': server_info.get('state')})
            self._result['diff']['after'].update({'state': target_state})
        if not self._module.check_mode:
            self._post('servers/%s/%s' % (server_info['uuid'], actions[target_state]))
            server_info = self._wait_for_state((target_state,))
    return server_info