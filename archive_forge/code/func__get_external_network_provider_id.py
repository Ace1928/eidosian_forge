from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_external_network_provider_id(self, external_provider):
    return external_provider.get('id') or get_id_by_name(self._connection.system_service().openstack_network_providers_service(), external_provider.get('name'))