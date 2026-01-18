from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_firewall_rules(module, oneandone_conn, firewall_id, rules):
    """
    Adds new rules to a firewall policy.
    """
    try:
        firewall_rules = []
        for rule in rules:
            firewall_rule = oneandone.client.FirewallPolicyRule(protocol=rule['protocol'], port_from=rule['port_from'], port_to=rule['port_to'], source=rule['source'])
            firewall_rules.append(firewall_rule)
        if module.check_mode:
            firewall_policy_id = get_firewall_policy(oneandone_conn, firewall_id)
            if firewall_rules and firewall_policy_id:
                return True
            return False
        firewall_policy = oneandone_conn.add_firewall_policy_rule(firewall_id=firewall_id, firewall_policy_rules=firewall_rules)
        return firewall_policy
    except Exception as e:
        module.fail_json(msg=str(e))