from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
from ansible.module_utils._text import to_native
class VmwareHostSnmp(PyVmomi):
    """Manage SNMP configuration for an ESXi host system"""

    def __init__(self, module):
        super(VmwareHostSnmp, self).__init__(module)
        self.results = {'changed': False}
        self.changed = False
        self.hosts = None
        if self.is_vcenter():
            esxi_hostname = self.params['esxi_hostname']
            cluster_name = self.params['cluster_name']
            if all([esxi_hostname, cluster_name]):
                self.module.fail_json(msg='Only one of parameters [cluster_name, esxi_hostname] is required')
            else:
                if cluster_name:
                    self.hosts = self.get_all_host_objs(cluster_name=cluster_name)
                    if self.hosts is None:
                        self.module.fail_json(msg='Failed to find host system in cluster %s.' % cluster_name)
                if esxi_hostname:
                    self.hosts = self.get_all_host_objs(esxi_host_name=esxi_hostname)
                    if self.hosts is None:
                        self.module.fail_json(msg='Failed to find host system.')
        else:
            host = find_obj(self.content, [vim.HostSystem], None)
            self.hosts = list()
            self.hosts.append(host)

    def ensure(self):
        """Manage SNMP configuration for an ESXi host system"""
        snmp_state = self.params.get('state')
        snmp_port = self.params.get('snmp_port')
        community = self.params.get('community')
        desired_trap_targets = self.params.get('trap_targets')
        hw_source = self.params.get('hw_source')
        log_level = self.params.get('log_level')
        send_trap = self.params.get('send_trap')
        trap_filter = self.params.get('trap_filter')
        sys_contact = self.params.get('sys_contact')
        sys_location = self.params.get('sys_location')
        event_filter = None
        if trap_filter:
            event_filter = ';'.join(trap_filter)
        reset_hint = None
        for host in self.hosts:
            changed = False
            changed_list = []
            self.results[host.name] = dict(changed=False, msg='')
            snmp_system = host.configManager.snmpSystem
            if snmp_system:
                if snmp_system.configuration:
                    snmp_config_spec = snmp_system.configuration
                else:
                    self.module.fail_json(msg="SNMP agent configuration isn't supported on the ESXi host")
            else:
                self.module.fail_json(msg="SNMP system isn't available on the ESXi host")
            self.results[host.name]['state'] = snmp_state
            if snmp_state == 'reset':
                changed = True
                if snmp_config_spec.enabled:
                    self.results[host.name]['state_previous'] = 'enabled'
                else:
                    self.results[host.name]['state_previous'] = 'disabled'
                self.results[host.name]['port_previous'] = snmp_config_spec.port
                self.results[host.name]['community_previous'] = snmp_config_spec.readOnlyCommunities
                self.results[host.name]['trap_targets_previous'] = self.get_previous_targets(snmp_config_spec.trapTargets)
                for option in snmp_config_spec.option:
                    if option.key == 'EnvEventSource' and option.value != hw_source:
                        self.results[host.name]['hw_source_previous'] = option.value
                    if option.key == 'loglevel' and option.value != hw_source:
                        self.results[host.name]['log_level_previous'] = option.value
                    if option.key == 'EventFilter' and option.value != hw_source:
                        self.results[host.name]['trap_filter_previous'] = option.value.split(';')
                    if option.key == 'syscontact' and option.value != hw_source:
                        self.results[host.name]['syscontact_previous'] = option.value
                    if option.key == 'syslocation' and option.value != hw_source:
                        self.results[host.name]['syslocation_previous'] = option.value
                destination = vim.host.SnmpSystem.SnmpConfigSpec.Destination()
                destination.hostName = ''
                destination.port = 0
                destination.community = ''
                options = []
                options.append(self.create_option('EnvEventSource', 'indications'))
                options.append(self.create_option('EventFilter', 'reset'))
                snmp_config_spec = vim.host.SnmpSystem.SnmpConfigSpec()
                snmp_config_spec.readOnlyCommunities = ['']
                snmp_config_spec.trapTargets = [destination]
                snmp_config_spec.port = 161
                snmp_config_spec.enabled = False
                snmp_config_spec.option = options
            else:
                if snmp_state == 'enabled' and (not snmp_config_spec.enabled):
                    changed = True
                    changed_list.append('state')
                    self.results[host.name]['state_previous'] = 'disabled'
                    snmp_config_spec.enabled = True
                elif snmp_state == 'disabled' and snmp_config_spec.enabled:
                    changed = True
                    changed_list.append('state')
                    self.results[host.name]['state_previous'] = 'enabled'
                    snmp_config_spec.enabled = False
                self.results[host.name]['port'] = snmp_port
                if snmp_config_spec.port != snmp_port:
                    changed = True
                    changed_list.append('port')
                    self.results[host.name]['port_previous'] = snmp_config_spec.port
                    snmp_config_spec.port = snmp_port
                self.results[host.name]['community'] = community
                if snmp_config_spec.readOnlyCommunities != community:
                    changed = True
                    changed_list.append('community list')
                    self.results[host.name]['community_previous'] = snmp_config_spec.readOnlyCommunities
                    if community:
                        snmp_config_spec.readOnlyCommunities = community
                    else:
                        reset_hint = True
                self.results[host.name]['trap_targets'] = desired_trap_targets
                if snmp_config_spec.trapTargets:
                    if desired_trap_targets:
                        temp_desired_targets = []
                        for target in desired_trap_targets:
                            dest_hostname, dest_port, dest_community = self.check_if_options_are_valid(target)
                            trap_target_found = False
                            for trap_target in snmp_config_spec.trapTargets:
                                if trap_target.hostName == dest_hostname:
                                    if trap_target.port != dest_port or trap_target.community != dest_community:
                                        changed = True
                                        changed_list.append("trap target '%s'" % dest_hostname)
                                    trap_target_found = True
                                    break
                            if not trap_target_found:
                                changed = True
                                changed_list.append("trap target '%s'" % dest_hostname)
                            destination = self.build_destination(dest_hostname, dest_port, dest_community)
                            temp_desired_targets.append(destination)
                        for trap_target in snmp_config_spec.trapTargets:
                            target_found = False
                            for target in desired_trap_targets:
                                if trap_target.hostName == target.get('hostname'):
                                    target_found = True
                                    break
                            if not target_found:
                                changed = True
                                changed_list.append("trap target '%s'" % trap_target.hostName)
                        if changed:
                            self.results[host.name]['trap_targets_previous'] = self.get_previous_targets(snmp_config_spec.trapTargets)
                            snmp_config_spec.trapTargets = temp_desired_targets
                    else:
                        changed = True
                        changed_list.append('trap targets')
                        self.results[host.name]['trap_targets_previous'] = self.get_previous_targets(snmp_config_spec.trapTargets)
                        reset_hint = True
                elif desired_trap_targets:
                    changed = True
                    changed_list.append('trap targets')
                    self.results[host.name]['trap_targets_previous'] = None
                    desired_targets = []
                    for target in desired_trap_targets:
                        dest_hostname, dest_port, dest_community = self.check_if_options_are_valid(target)
                        destination = self.build_destination(dest_hostname, dest_port, dest_community)
                        desired_targets.append(destination)
                    snmp_config_spec.trapTargets = desired_targets
                self.results[host.name]['hw_source'] = hw_source
                self.results[host.name]['log_level'] = log_level
                self.results[host.name]['trap_filter'] = trap_filter
                event_filter_found = False
                sys_contact_found = False
                sys_location_found = False
                if snmp_config_spec.option:
                    for option in snmp_config_spec.option:
                        if option.key == 'EnvEventSource' and option.value != hw_source:
                            changed = True
                            changed_list.append('HW source')
                            self.results[host.name]['hw_source_previous'] = option.value
                            option.value = hw_source
                        if option.key == 'loglevel' and option.value != log_level:
                            changed = True
                            changed_list.append('log level')
                            self.results[host.name]['log_level_previous'] = option.value
                            option.value = log_level
                        if option.key == 'EventFilter':
                            event_filter_found = True
                            if event_filter and option.value != event_filter:
                                changed = True
                                changed_list.append('trap filter')
                                self.results[host.name]['trap_filter_previous'] = option.value.split(';')
                                option.value = event_filter
                        if option.key == 'syscontact':
                            sys_contact_found = True
                            if sys_contact is not None and option.value != sys_contact:
                                changed = True
                                changed_list.append('sys contact')
                                self.results[host.name]['sys_contact_previous'] = option.value
                                option.value = sys_contact
                        if option.key == 'syslocation':
                            sys_location_found = True
                            if sys_location is not None and option.value != sys_location:
                                changed = True
                                changed_list.append('sys location')
                                self.results[host.name]['sys_location_previous'] = option.value
                                option.value = sys_location
                if trap_filter and (not event_filter_found):
                    changed = True
                    changed_list.append('trap filter')
                    self.results[host.name]['trap_filter_previous'] = []
                    snmp_config_spec.option.append(self.create_option('EventFilter', event_filter))
                elif not trap_filter and event_filter_found:
                    changed = True
                    changed_list.append('trap filter')
                    for option in snmp_config_spec.option:
                        if option.key == 'EventFilter':
                            self.results[host.name]['trap_filter_previous'] = option.value.split(';')
                    reset_hint = True
                if sys_contact and (not sys_contact_found):
                    changed = True
                    changed_list.append('sys contact')
                    self.results[host.name]['sys_contact_previous'] = ''
                    snmp_config_spec.option.append(self.create_option('syscontact', sys_contact))
                if sys_location and (not sys_location_found):
                    changed = True
                    changed_list.append('sys location')
                    self.results[host.name]['sys_location_previous'] = ''
                    snmp_config_spec.option.append(self.create_option('syslocation', sys_location))
            if changed:
                if snmp_state == 'reset':
                    if self.module.check_mode:
                        message = 'SNMP agent would be reset to factory defaults'
                    else:
                        message = 'SNMP agent config reset to factory defaults'
                else:
                    if self.module.check_mode:
                        changed_suffix = ' would be changed'
                    else:
                        changed_suffix = ' changed'
                    if len(changed_list) > 2:
                        message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
                    elif len(changed_list) == 2:
                        message = ' and '.join(changed_list)
                    elif len(changed_list) == 1:
                        message = changed_list[0]
                    message = 'SNMP ' + message + changed_suffix
                if reset_hint:
                    message += '. Agent reset required!'
                if not self.module.check_mode:
                    try:
                        snmp_system.ReconfigureSnmpAgent(snmp_config_spec)
                    except vim.fault.NotFound as not_found:
                        self.module.fail_json(msg='Not found : %s' % to_native(not_found))
                    except vim.fault.InsufficientResourcesFault as insufficient_resources:
                        self.module.fail_json(msg='Insufficient resources : %s' % to_native(insufficient_resources))
            else:
                message = 'SNMP already configured properly'
            if not snmp_state == 'reset' and send_trap and desired_trap_targets:
                if changed:
                    message += ' and '
                else:
                    message += ', but '
                changed = True
                if self.module.check_mode:
                    message = message + 'a test trap would be sent'
                else:
                    try:
                        snmp_system.SendTestNotification()
                        message = message + 'a test trap was sent'
                    except vim.fault.NotFound as not_found:
                        self.module.fail_json(msg='Error during trap test : Not found : %s' % to_native(not_found))
                    except vim.fault.InsufficientResourcesFault as insufficient_resources:
                        self.module.fail_json(msg='Error during trap test : Insufficient resources : %s' % to_native(insufficient_resources))
            self.changed = any([self.changed, changed])
            self.results[host.name]['changed'] = changed
            self.results[host.name]['msg'] = message
        self.results['changed'] = self.changed
        self.module.exit_json(**self.results)

    @staticmethod
    def create_option(key, value):
        """Create option"""
        option = vim.KeyValue()
        option.key = key
        option.value = value
        return option

    @staticmethod
    def get_previous_targets(trap_targets):
        """Get target entries from trap targets object"""
        previous_targets = []
        for target in trap_targets:
            temp = dict()
            temp['hostname'] = target.hostName
            temp['port'] = target.port
            temp['community'] = target.community
            previous_targets.append(temp)
        return previous_targets

    @staticmethod
    def build_destination(dest_hostname, dest_port, dest_community):
        """Build destination spec"""
        destination = vim.host.SnmpSystem.SnmpConfigSpec.Destination()
        destination.hostName = dest_hostname
        destination.port = dest_port
        destination.community = dest_community
        return destination

    def check_if_options_are_valid(self, target):
        """Check if options are valid"""
        dest_hostname = target.get('hostname', None)
        if dest_hostname is None:
            self.module.fail_json(msg="Please specify hostname for the trap target as it's a required parameter")
        dest_port = target.get('port', None)
        if dest_port is None:
            self.module.fail_json(msg="Please specify port for the trap target as it's a required parameter")
        dest_community = target.get('community', None)
        if dest_community is None:
            self.module.fail_json(msg="Please specify community for the trap target as it's a required parameter")
        return (dest_hostname, dest_port, dest_community)