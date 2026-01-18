from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_external_network_providers_entity(self):
    if self.param('external_network_providers') is not None:
        return [otypes.ExternalProvider(id=self._get_external_network_provider_id(external_provider)) for external_provider in self.param('external_network_providers')]