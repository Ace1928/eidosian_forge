from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def absent_server(self):
    server_info = self._get_server_info()
    if server_info.get('state') != 'absent':
        self._result['changed'] = True
        self._result['diff']['before'] = deepcopy(server_info)
        self._result['diff']['after'] = self._init_server_container()
        if not self._module.check_mode:
            self._delete('servers/%s' % server_info['uuid'])
            server_info = self._wait_for_state(('absent',))
    return server_info