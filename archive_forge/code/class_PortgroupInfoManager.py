from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class PortgroupInfoManager(PyVmomi):
    """Class to manage Port Group info"""

    def __init__(self, module):
        super(PortgroupInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')
        self.policies = self.params.get('policies')

    @staticmethod
    def normalize_pg_info(portgroup_obj, policy_info):
        """Create Port Group information"""
        pg_info_dict = dict()
        spec = portgroup_obj.spec
        pg_info_dict['portgroup'] = spec.name
        pg_info_dict['vlan_id'] = spec.vlanId
        pg_info_dict['vswitch'] = spec.vswitchName
        if policy_info:
            if spec.policy.security:
                promiscuous_mode = spec.policy.security.allowPromiscuous
                mac_changes = spec.policy.security.macChanges
                forged_transmits = spec.policy.security.forgedTransmits
                pg_info_dict['security'] = ['No override' if promiscuous_mode is None else promiscuous_mode, 'No override' if mac_changes is None else mac_changes, 'No override' if forged_transmits is None else forged_transmits]
            else:
                pg_info_dict['security'] = ['No override', 'No override', 'No override']
            if spec.policy.shapingPolicy and spec.policy.shapingPolicy.enabled is not None:
                pg_info_dict['ts'] = portgroup_obj.spec.policy.shapingPolicy.enabled
            else:
                pg_info_dict['ts'] = 'No override'
            if spec.policy.nicTeaming:
                if spec.policy.nicTeaming.policy is None:
                    pg_info_dict['lb'] = 'No override'
                else:
                    pg_info_dict['lb'] = spec.policy.nicTeaming.policy
                if spec.policy.nicTeaming.notifySwitches is None:
                    pg_info_dict['notify'] = 'No override'
                else:
                    pg_info_dict['notify'] = spec.policy.nicTeaming.notifySwitches
                if spec.policy.nicTeaming.rollingOrder is None:
                    pg_info_dict['failback'] = 'No override'
                else:
                    pg_info_dict['failback'] = not spec.policy.nicTeaming.rollingOrder
                if spec.policy.nicTeaming.nicOrder is None:
                    pg_info_dict['failover_active'] = 'No override'
                    pg_info_dict['failover_standby'] = 'No override'
                else:
                    pg_info_dict['failover_active'] = spec.policy.nicTeaming.nicOrder.activeNic
                    pg_info_dict['failover_standby'] = spec.policy.nicTeaming.nicOrder.standbyNic
                if spec.policy.nicTeaming.failureCriteria is None:
                    pg_info_dict['failure_detection'] = 'No override'
                elif spec.policy.nicTeaming.failureCriteria.checkBeacon:
                    pg_info_dict['failure_detection'] = 'beacon_probing'
                else:
                    pg_info_dict['failure_detection'] = 'link_status_only'
            else:
                pg_info_dict['lb'] = 'No override'
                pg_info_dict['notify'] = 'No override'
                pg_info_dict['failback'] = 'No override'
                pg_info_dict['failover_active'] = 'No override'
                pg_info_dict['failover_standby'] = 'No override'
                pg_info_dict['failure_detection'] = 'No override'
        return pg_info_dict

    def gather_host_portgroup_info(self):
        """Gather Port Group info per ESXi host"""
        hosts_pg_info = dict()
        for host in self.hosts:
            pgs = host.config.network.portgroup
            hosts_pg_info[host.name] = []
            for portgroup in pgs:
                hosts_pg_info[host.name].append(self.normalize_pg_info(portgroup_obj=portgroup, policy_info=self.policies))
        return hosts_pg_info