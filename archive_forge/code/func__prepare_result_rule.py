from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def _prepare_result_rule(self, rule: FirewallRule):
    return {'direction': to_native(rule.direction), 'protocol': to_native(rule.protocol), 'port': to_native(rule.port) if rule.port is not None else None, 'source_ips': [to_native(cidr) for cidr in rule.source_ips], 'destination_ips': [to_native(cidr) for cidr in rule.destination_ips], 'description': to_native(rule.description) if rule.description is not None else None}