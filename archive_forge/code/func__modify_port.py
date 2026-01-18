from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _modify_port(module, oneandone_conn, monitoring_policy_id, port_id, port):
    """
    Modifies a monitoring policy port.
    """
    try:
        if module.check_mode:
            cm_port = oneandone_conn.get_monitoring_policy_port(monitoring_policy_id=monitoring_policy_id, port_id=port_id)
            if cm_port:
                return True
            return False
        monitoring_policy_port = oneandone.client.Port(protocol=port['protocol'], port=port['port'], alert_if=port['alert_if'], email_notification=port['email_notification'])
        monitoring_policy = oneandone_conn.modify_port(monitoring_policy_id=monitoring_policy_id, port_id=port_id, port=monitoring_policy_port)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))