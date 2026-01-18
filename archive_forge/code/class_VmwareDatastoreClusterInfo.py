from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareDatastoreClusterInfo(PyVmomi):

    def __init__(self, module):
        super(VmwareDatastoreClusterInfo, self).__init__(module)
        self.module = module
        self.params = module.params
        datacenter_name = self.params.get('datacenter')
        datacenter_obj = self.find_datacenter_by_name(datacenter_name)
        if datacenter_obj is None:
            self.module.fail_json(msg='Unable to find datacenter with name %s' % datacenter_name)
        datastore_cluster_name = self.params.get('datastore_cluster')
        datastore_cluster_obj = self.find_datastore_cluster_by_name(datastore_cluster_name, datacenter=datacenter_obj)
        datastore_name = self.get_recommended_datastore(datastore_cluster_obj=datastore_cluster_obj)
        if not datastore_name:
            datastore_name = ''
        result = dict(changed=False, recommended_datastore=datastore_name)
        self.module.exit_json(**result)