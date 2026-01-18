from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VswitchInfoManager(PyVmomi):
    """Class to gather vSwitch info"""

    def __init__(self, module):
        super(VswitchInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')
        self.policies = self.params.get('policies')

    @staticmethod
    def serialize_pnics(vswitch_obj):
        """Get pnic names"""
        pnics = []
        for pnic in vswitch_obj.pnic:
            pnics.append(pnic.split('-', 3)[-1])
        return pnics

    @staticmethod
    def normalize_vswitch_info(vswitch_obj, policy_info):
        """Create vSwitch information"""
        vswitch_info_dict = dict()
        spec = vswitch_obj.spec
        vswitch_info_dict['pnics'] = VswitchInfoManager.serialize_pnics(vswitch_obj)
        vswitch_info_dict['mtu'] = vswitch_obj.mtu
        vswitch_info_dict['num_ports'] = spec.numPorts
        if policy_info:
            if spec.policy.security:
                vswitch_info_dict['security'] = [spec.policy.security.allowPromiscuous, spec.policy.security.macChanges, spec.policy.security.forgedTransmits]
            if spec.policy.shapingPolicy:
                vswitch_info_dict['ts'] = spec.policy.shapingPolicy.enabled
            if spec.policy.nicTeaming:
                vswitch_info_dict['lb'] = spec.policy.nicTeaming.policy
                vswitch_info_dict['notify'] = spec.policy.nicTeaming.notifySwitches
                vswitch_info_dict['failback'] = not spec.policy.nicTeaming.rollingOrder
                vswitch_info_dict['failover_active'] = spec.policy.nicTeaming.nicOrder.activeNic
                vswitch_info_dict['failover_standby'] = spec.policy.nicTeaming.nicOrder.standbyNic
                if spec.policy.nicTeaming.failureCriteria.checkBeacon:
                    vswitch_info_dict['failure_detection'] = 'beacon_probing'
                else:
                    vswitch_info_dict['failure_detection'] = 'link_status_only'
        return vswitch_info_dict

    def gather_vswitch_info(self):
        """Gather vSwitch info"""
        hosts_vswitch_info = dict()
        for host in self.hosts:
            network_manager = host.configManager.networkSystem
            if network_manager:
                temp_switch_dict = dict()
                for vswitch in network_manager.networkInfo.vswitch:
                    temp_switch_dict[vswitch.name] = self.normalize_vswitch_info(vswitch_obj=vswitch, policy_info=self.policies)
                hosts_vswitch_info[host.name] = temp_switch_dict
        return hosts_vswitch_info