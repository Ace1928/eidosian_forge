from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
import socket
class VmwareFirewallManager(PyVmomi):

    def __init__(self, module):
        super(VmwareFirewallManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.options = self.params.get('options', dict())
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        self.firewall_facts = dict()
        self.rule_options = self.module.params.get('rules')
        self.gather_rule_set()

    def gather_rule_set(self):
        for host in self.hosts:
            self.firewall_facts[host.name] = {}
            firewall_system = host.configManager.firewallSystem
            if firewall_system:
                for rule_set_obj in firewall_system.firewallInfo.ruleset:
                    temp_rule_dict = dict()
                    temp_rule_dict['enabled'] = rule_set_obj.enabled
                    allowed_host = rule_set_obj.allowedHosts
                    rule_allow_host = dict()
                    rule_allow_host['ip_address'] = allowed_host.ipAddress
                    rule_allow_host['ip_network'] = [ip.network + '/' + str(ip.prefixLength) for ip in allowed_host.ipNetwork]
                    rule_allow_host['all_ip'] = allowed_host.allIp
                    temp_rule_dict['allowed_hosts'] = rule_allow_host
                    self.firewall_facts[host.name][rule_set_obj.key] = temp_rule_dict

    def check_params(self):
        rules_by_host = {}
        for host in self.hosts:
            rules_by_host[host.name] = self.firewall_facts[host.name].keys()
        for rule_option in self.rule_options:
            rule_name = rule_option.get('name')
            hosts_with_rule_name = [h for h, r in rules_by_host.items() if rule_name in r]
            hosts_without_rule_name = set([i.name for i in self.hosts]) - set(hosts_with_rule_name)
            if hosts_without_rule_name:
                self.module.fail_json(msg="rule named '%s' wasn't found on hosts: %s" % (rule_name, hosts_without_rule_name))
            allowed_hosts = rule_option.get('allowed_hosts')
            if allowed_hosts is not None:
                for ip_address in allowed_hosts.get('ip_address'):
                    try:
                        is_ipaddress(ip_address)
                    except ValueError:
                        self.module.fail_json(msg='The provided IP address %s is not a valid IP for the rule %s' % (ip_address, rule_name))
                for ip_network in allowed_hosts.get('ip_network'):
                    try:
                        is_ipaddress(ip_network)
                    except ValueError:
                        self.module.fail_json(msg='The provided IP network %s is not a valid network for the rule %s' % (ip_network, rule_name))

    def ensure(self):
        """
        Function to ensure rule set configuration

        """
        fw_change_list = []
        enable_disable_changed = False
        allowed_ip_changed = False
        results = dict(changed=False, rule_set_state=dict())
        for host in self.hosts:
            firewall_system = host.configManager.firewallSystem
            if firewall_system is None:
                continue
            results['rule_set_state'][host.name] = {}
            for rule_option in self.rule_options:
                rule_name = rule_option.get('name', None)
                current_rule_state = self.firewall_facts[host.name][rule_name]['enabled']
                if current_rule_state != rule_option['enabled']:
                    try:
                        if not self.module.check_mode:
                            if rule_option['enabled']:
                                firewall_system.EnableRuleset(id=rule_name)
                            else:
                                firewall_system.DisableRuleset(id=rule_name)
                        enable_disable_changed = True
                    except vim.fault.NotFound as not_found:
                        self.module.fail_json(msg='Failed to enable rule set %s as rule set id is unknown : %s' % (rule_name, to_native(not_found.msg)))
                    except vim.fault.HostConfigFault as host_config_fault:
                        self.module.fail_json(msg='Failed to enabled rule set %s as an internal error happened while reconfiguring rule set : %s' % (rule_name, to_native(host_config_fault.msg)))
                permitted_networking = self.firewall_facts[host.name][rule_name]
                rule_allows_all = permitted_networking['allowed_hosts']['all_ip']
                rule_allowed_ips = set(permitted_networking['allowed_hosts']['ip_address'])
                rule_allowed_networks = set(permitted_networking['allowed_hosts']['ip_network'])
                allowed_hosts = rule_option.get('allowed_hosts')
                playbook_allows_all = False if allowed_hosts is None else allowed_hosts.get('all_ip')
                playbook_allowed_ips = set([]) if allowed_hosts is None else set(allowed_hosts.get('ip_address'))
                playbook_allowed_networks = set([]) if allowed_hosts is None else set(allowed_hosts.get('ip_network'))
                allowed_all_ips_different = bool(rule_allows_all != playbook_allows_all)
                ip_list_different = bool(rule_allowed_ips != playbook_allowed_ips)
                ip_network_different = bool(rule_allowed_networks != playbook_allowed_networks)
                if allowed_all_ips_different is True or ip_list_different is True or ip_network_different is True:
                    try:
                        allowed_ip_changed = True
                        if not self.module.check_mode:
                            firewall_spec = vim.host.Ruleset.RulesetSpec()
                            firewall_spec.allowedHosts = vim.host.Ruleset.IpList()
                            firewall_spec.allowedHosts.allIp = playbook_allows_all
                            firewall_spec.allowedHosts.ipAddress = list(playbook_allowed_ips)
                            firewall_spec.allowedHosts.ipNetwork = []
                            for i in playbook_allowed_networks:
                                address, mask = i.split('/')
                                tmp_ip_network_spec = vim.host.Ruleset.IpNetwork()
                                tmp_ip_network_spec.network = address
                                tmp_ip_network_spec.prefixLength = int(mask)
                                firewall_spec.allowedHosts.ipNetwork.append(tmp_ip_network_spec)
                            firewall_system.UpdateRuleset(id=rule_name, spec=firewall_spec)
                    except vim.fault.NotFound as not_found:
                        self.module.fail_json(msg='Failed to configure rule set %s as rule set id is unknown : %s' % (rule_name, to_native(not_found.msg)))
                    except vim.fault.HostConfigFault as host_config_fault:
                        self.module.fail_json(msg='Failed to configure rule set %s as an internal error happened while reconfiguring rule set : %s' % (rule_name, to_native(host_config_fault.msg)))
                    except vim.fault.RuntimeFault as runtime_fault:
                        self.module.fail_json(msg='Failed to configure the rule set %s as a runtime error happened while applying the reconfiguration: %s' % (rule_name, to_native(runtime_fault.msg)))
                results['rule_set_state'][host.name][rule_name] = {'current_state': rule_option['enabled'], 'previous_state': current_rule_state, 'desired_state': rule_option['enabled'], 'allowed_hosts': {'current_allowed_all': playbook_allows_all, 'previous_allowed_all': permitted_networking['allowed_hosts']['all_ip'], 'desired_allowed_all': playbook_allows_all, 'current_allowed_ip': playbook_allowed_ips, 'previous_allowed_ip': set(permitted_networking['allowed_hosts']['ip_address']), 'desired_allowed_ip': playbook_allowed_ips, 'current_allowed_networks': playbook_allowed_networks, 'previous_allowed_networks': set(permitted_networking['allowed_hosts']['ip_network']), 'desired_allowed_networks': playbook_allowed_networks}}
        if enable_disable_changed or allowed_ip_changed:
            fw_change_list.append(True)
        if any(fw_change_list):
            results['changed'] = True
        self.module.exit_json(**results)