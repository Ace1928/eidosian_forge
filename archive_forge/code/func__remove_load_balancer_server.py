from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _remove_load_balancer_server(module, oneandone_conn, load_balancer_id, server_ip_id):
    """
    Unassigns a server/IP from a load balancer.
    """
    try:
        if module.check_mode:
            lb_server = oneandone_conn.get_load_balancer_server(load_balancer_id=load_balancer_id, server_ip_id=server_ip_id)
            if lb_server:
                return True
            return False
        load_balancer = oneandone_conn.remove_load_balancer_server(load_balancer_id=load_balancer_id, server_ip_id=server_ip_id)
        return load_balancer
    except Exception as ex:
        module.fail_json(msg=str(ex))