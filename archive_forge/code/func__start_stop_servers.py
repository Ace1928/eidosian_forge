from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _start_stop_servers(module, clc, server_ids):
    """
        Start or Stop the servers on the provided list
        :param module: the AnsibleModule object
        :param clc: the clc-sdk instance to use
        :param server_ids: list of servers to start or stop
        :return: a list of dictionaries with server information about the servers that were started or stopped
        """
    p = module.params
    state = p.get('state')
    changed = False
    changed_servers = []
    server_dict_array = []
    result_server_ids = []
    request_list = []
    if not isinstance(server_ids, list) or len(server_ids) < 1:
        return module.fail_json(msg='server_ids should be a list of servers, aborting')
    servers = clc.v2.Servers(server_ids).Servers()
    for server in servers:
        if server.powerState != state:
            changed_servers.append(server)
            if not module.check_mode:
                request_list.append(ClcServer._change_server_power_state(module, server, state))
            changed = True
    ClcServer._wait_for_requests(module, request_list)
    ClcServer._refresh_servers(module, changed_servers)
    for server in set(changed_servers + servers):
        try:
            server.data['ipaddress'] = server.details['ipAddresses'][0]['internal']
            server.data['publicip'] = str(server.PublicIPs().public_ips[0])
        except (KeyError, IndexError):
            pass
        server_dict_array.append(server.data)
        result_server_ids.append(server.id)
    return (changed, server_dict_array, result_server_ids)