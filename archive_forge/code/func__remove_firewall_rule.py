from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _remove_firewall_rule(module, oneandone_conn, firewall_id, rule_id):
    """
    Removes a rule from a firewall policy.
    """
    try:
        if module.check_mode:
            rule = oneandone_conn.get_firewall_policy_rule(firewall_id=firewall_id, rule_id=rule_id)
            if rule:
                return True
            return False
        firewall_policy = oneandone_conn.remove_firewall_rule(firewall_id=firewall_id, rule_id=rule_id)
        return firewall_policy
    except Exception as e:
        module.fail_json(msg=str(e))