from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VMwareHostSslManager(PyVmomi):

    def __init__(self, module):
        super(VMwareHostSslManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        self.hosts_info = {}

    def gather_ssl_info(self):
        for host in self.hosts:
            self.hosts_info[host.name] = dict(principal='', owner_tag='', ssl_thumbprints=[])
            host_ssl_info_mgr = host.config.sslThumbprintInfo
            if host_ssl_info_mgr:
                self.hosts_info[host.name]['principal'] = host_ssl_info_mgr.principal
                self.hosts_info[host.name]['owner_tag'] = host_ssl_info_mgr.ownerTag
                self.hosts_info[host.name]['ssl_thumbprints'] = list(host_ssl_info_mgr.sslThumbprints)
        self.module.exit_json(changed=False, host_ssl_info=self.hosts_info)