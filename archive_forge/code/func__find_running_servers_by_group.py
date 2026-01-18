from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_running_servers_by_group(module, datacenter, count_group):
    """
        Find a list of running servers in the provided group
        :param module: the AnsibleModule object
        :param datacenter: the clc-sdk.Datacenter instance to use to lookup the group
        :param count_group: the group to count the servers
        :return: list of servers, and list of running servers
        """
    group = ClcServer._find_group(module=module, datacenter=datacenter, lookup_group=count_group)
    servers = group.Servers().Servers()
    running_servers = []
    for server in servers:
        if server.status == 'active' and server.powerState == 'started':
            running_servers.append(server)
    return (servers, running_servers)