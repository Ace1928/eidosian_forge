from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VmwareConfigInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmwareConfigInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)

    def gather_host_info(self):
        hosts_info = {}
        for host in self.hosts:
            host_info = {}
            for option in host.configManager.advancedOption.QueryOptions():
                host_info[option.key] = option.value
            hosts_info[host.name] = host_info
        return hosts_info