from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def _diff_firewall_resources(self, operator) -> list[FirewallResource]:
    before = self._prepare_result()
    resources: list[FirewallResource] = []
    servers: list[str] | None = self.module.params.get('servers')
    if servers:
        for server_param in servers:
            try:
                server: BoundServer = self._client_get_by_name_or_id('servers', server_param)
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
            if operator(server.name, before['servers']):
                resources.append(FirewallResource(type=FirewallResource.TYPE_SERVER, server=server))
    label_selectors = self.module.params.get('label_selectors')
    if label_selectors:
        for label_selector in label_selectors:
            if operator(label_selector, before['label_selectors']):
                resources.append(FirewallResource(type=FirewallResource.TYPE_LABEL_SELECTOR, label_selector=FirewallResourceLabelSelector(selector=label_selector)))
    return resources