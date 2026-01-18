from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _modify_servers(self, server_ids):
    """
        modify the servers configuration on the provided list
        :param server_ids: list of servers to modify
        :return: a list of dictionaries with server information about the servers that were modified
        """
    p = self.module.params
    state = p.get('state')
    server_params = {'cpu': p.get('cpu'), 'memory': p.get('memory'), 'anti_affinity_policy_id': p.get('anti_affinity_policy_id'), 'anti_affinity_policy_name': p.get('anti_affinity_policy_name'), 'alert_policy_id': p.get('alert_policy_id'), 'alert_policy_name': p.get('alert_policy_name')}
    changed = False
    server_changed = False
    aa_changed = False
    ap_changed = False
    server_dict_array = []
    result_server_ids = []
    request_list = []
    changed_servers = []
    if not isinstance(server_ids, list) or len(server_ids) < 1:
        return self.module.fail_json(msg='server_ids should be a list of servers, aborting')
    servers = self._get_servers_from_clc(server_ids, 'Failed to obtain server list from the CLC API')
    for server in servers:
        if state == 'present':
            server_changed, server_result = self._ensure_server_config(server, server_params)
            if server_result:
                request_list.append(server_result)
            aa_changed = self._ensure_aa_policy_present(server, server_params)
            ap_changed = self._ensure_alert_policy_present(server, server_params)
        elif state == 'absent':
            aa_changed = self._ensure_aa_policy_absent(server, server_params)
            ap_changed = self._ensure_alert_policy_absent(server, server_params)
        if server_changed or aa_changed or ap_changed:
            changed_servers.append(server)
            changed = True
    self._wait_for_requests(self.module, request_list)
    self._refresh_servers(self.module, changed_servers)
    for server in changed_servers:
        server_dict_array.append(server.data)
        result_server_ids.append(server.id)
    return (changed, server_dict_array, result_server_ids)