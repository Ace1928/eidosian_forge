from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _add_public_ip_to_servers(module, should_add_public_ip, servers, public_ip_protocol, public_ip_ports):
    """
        Create a public IP for servers
        :param module: the AnsibleModule object
        :param should_add_public_ip: boolean - whether or not to provision a public ip for servers.  Skipped if False
        :param servers: List of servers to add public ips to
        :param public_ip_protocol: a protocol to allow for the public ips
        :param public_ip_ports: list of ports to allow for the public ips
        :return: none
        """
    failed_servers = []
    if not should_add_public_ip:
        return failed_servers
    ports_lst = []
    request_list = []
    server = None
    for port in public_ip_ports:
        ports_lst.append({'protocol': public_ip_protocol, 'port': port})
    try:
        if not module.check_mode:
            for server in servers:
                request = server.PublicIPs().Add(ports_lst)
                request_list.append(request)
    except APIFailedResponse:
        failed_servers.append(server)
    ClcServer._wait_for_requests(module, request_list)
    return failed_servers