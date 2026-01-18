from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_processes(module, oneandone_conn, monitoring_policy_id, processes):
    """
    Adds new processes to a monitoring policy.
    """
    try:
        monitoring_policy_processes = []
        for _process in processes:
            monitoring_policy_process = oneandone.client.Process(process=_process['process'], alert_if=_process['alert_if'], email_notification=_process['email_notification'])
            monitoring_policy_processes.append(monitoring_policy_process)
        if module.check_mode:
            mp_id = get_monitoring_policy(oneandone_conn, monitoring_policy_id)
            if monitoring_policy_processes and mp_id:
                return True
            return False
        monitoring_policy = oneandone_conn.add_process(monitoring_policy_id=monitoring_policy_id, processes=monitoring_policy_processes)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))