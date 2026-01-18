from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class FeatureCapabilityInfoManager(PyVmomi):

    def __init__(self, module):
        super(FeatureCapabilityInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)

    def gather_host_feature_info(self):
        host_feature_info = dict()
        for host in self.hosts:
            host_feature_capabilities = host.config.featureCapability
            capability = []
            for fc in host_feature_capabilities:
                temp_dict = {'key': fc.key, 'feature_name': fc.featureName, 'value': fc.value}
                capability.append(temp_dict)
            host_feature_info[host.name] = capability
        return host_feature_info