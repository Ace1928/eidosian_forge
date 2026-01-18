from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def _update_server(self, server_info):
    previous_state = server_info.get('state')
    desired_server_group_ids = self._get_server_group_ids()
    if desired_server_group_ids is not None:
        current_server_group_ids = [grp['uuid'] for grp in server_info['server_groups']]
        if desired_server_group_ids != current_server_group_ids:
            self._module.warn('Server groups can not be mutated, server needs redeployment to change groups.')
    self.normalize_interfaces_param()
    wanted = self._module.params.get('interfaces')
    actual = server_info.get('interfaces')
    try:
        update_interfaces = not self.has_wanted_interfaces(wanted, actual)
    except KeyError as e:
        self._module.fail_json(msg="Error checking 'interfaces', missing key: %s" % e.args[0])
    if update_interfaces:
        server_info = self._update_param('interfaces', server_info)
        if not self._result['changed']:
            self._result['changed'] = server_info['interfaces'] != actual
    server_info = self._update_param('flavor', server_info, requires_stop=True)
    server_info = self._update_param('name', server_info)
    server_info = self._update_param('tags', server_info)
    if previous_state == 'running':
        server_info = self._start_stop_server(server_info, target_state='running', ignore_diff=True)
    return server_info