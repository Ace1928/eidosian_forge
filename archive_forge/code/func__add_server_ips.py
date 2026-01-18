from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_server_ips(module, oneandone_conn, firewall_id, server_ids):
    """
    Assigns servers to a firewall policy.
    """
    try:
        attach_servers = []
        for _server_id in server_ids:
            server = get_server(oneandone_conn, _server_id, True)
            attach_server = oneandone.client.AttachServer(server_id=server['id'], server_ip_id=next(iter(server['ips'] or []), None)['id'])
            attach_servers.append(attach_server)
        if module.check_mode:
            if attach_servers:
                return True
            return False
        firewall_policy = oneandone_conn.attach_server_firewall_policy(firewall_id=firewall_id, server_ips=attach_servers)
        return firewall_policy
    except Exception as e:
        module.fail_json(msg=str(e))