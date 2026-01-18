from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
from ansible.module_utils._text import to_native
def _mod_debug(self):
    if self.log_level == 'debug':
        self.result['debug'] = dict(datacenter_id=self._datacenter_id, datastore_id=self._datastore_id, library_item_id=self._library_item_id, folder_id=self._folder_id, host_id=self._host_id, cluster_id=self._cluster_id, resourcepool_id=self._resourcepool_id)