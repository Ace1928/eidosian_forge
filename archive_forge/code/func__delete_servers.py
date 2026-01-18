from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _delete_servers(module, clc, server_ids):
    """
        Delete the servers on the provided list
        :param module: the AnsibleModule object
        :param clc: the clc-sdk instance to use
        :param server_ids: list of servers to delete
        :return: a list of dictionaries with server information about the servers that were deleted
        """
    terminated_server_ids = []
    server_dict_array = []
    request_list = []
    if not isinstance(server_ids, list) or len(server_ids) < 1:
        return module.fail_json(msg='server_ids should be a list of servers, aborting')
    servers = clc.v2.Servers(server_ids).Servers()
    for server in servers:
        if not module.check_mode:
            request_list.append(server.Delete())
    ClcServer._wait_for_requests(module, request_list)
    for server in servers:
        terminated_server_ids.append(server.id)
    return (True, server_dict_array, terminated_server_ids)