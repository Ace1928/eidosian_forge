from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_hostsystem_by_name
class VcenterLicenseMgr(PyVmomi):

    def __init__(self, module):
        super(VcenterLicenseMgr, self).__init__(module)

    def find_key(self, licenses, license):
        for item in licenses:
            if item.licenseKey == license:
                return item
        return None

    def list_keys(self, licenses):
        keys = []
        for item in licenses:
            if item.used is None:
                continue
            keys.append(item.licenseKey)
        return keys