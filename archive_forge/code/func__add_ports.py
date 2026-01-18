from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_ports(module, oneandone_conn, monitoring_policy_id, ports):
    """
    Adds new ports to a monitoring policy.
    """
    try:
        monitoring_policy_ports = []
        for _port in ports:
            monitoring_policy_port = oneandone.client.Port(protocol=_port['protocol'], port=_port['port'], alert_if=_port['alert_if'], email_notification=_port['email_notification'])
            monitoring_policy_ports.append(monitoring_policy_port)
        if module.check_mode:
            if monitoring_policy_ports:
                return True
            return False
        monitoring_policy = oneandone_conn.add_port(monitoring_policy_id=monitoring_policy_id, ports=monitoring_policy_ports)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))