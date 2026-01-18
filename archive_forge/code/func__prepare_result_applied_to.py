from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def _prepare_result_applied_to(self, resource: FirewallResource):
    result = {'type': to_native(resource.type), 'server': to_native(resource.server.id) if resource.server is not None else None, 'label_selector': to_native(resource.label_selector.selector) if resource.label_selector is not None else None}
    if resource.applied_to_resources is not None:
        result['applied_to_resources'] = [{'type': to_native(item.type), 'server': to_native(item.server.id) if item.server is not None else None} for item in resource.applied_to_resources]
    return result