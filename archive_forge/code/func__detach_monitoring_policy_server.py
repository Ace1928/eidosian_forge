from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _detach_monitoring_policy_server(module, oneandone_conn, monitoring_policy_id, server_id):
    """
    Detaches a server from a monitoring policy.
    """
    try:
        if module.check_mode:
            mp_server = oneandone_conn.get_monitoring_policy_server(monitoring_policy_id=monitoring_policy_id, server_id=server_id)
            if mp_server:
                return True
            return False
        monitoring_policy = oneandone_conn.detach_monitoring_policy_server(monitoring_policy_id=monitoring_policy_id, server_id=server_id)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))