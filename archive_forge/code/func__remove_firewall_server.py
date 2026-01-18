from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _remove_firewall_server(module, oneandone_conn, firewall_id, server_ip_id):
    """
    Unassigns a server/IP from a firewall policy.
    """
    try:
        if module.check_mode:
            firewall_server = oneandone_conn.get_firewall_server(firewall_id=firewall_id, server_ip_id=server_ip_id)
            if firewall_server:
                return True
            return False
        firewall_policy = oneandone_conn.remove_firewall_server(firewall_id=firewall_id, server_ip_id=server_ip_id)
        return firewall_policy
    except Exception as e:
        module.fail_json(msg=str(e))