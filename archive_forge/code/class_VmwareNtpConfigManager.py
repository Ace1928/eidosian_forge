from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VmwareNtpConfigManager(PyVmomi):
    """Class to manage configured NTP servers"""

    def __init__(self, module):
        super(VmwareNtpConfigManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.ntp_servers = self.params.get('ntp_servers', list())
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system.')
        self.results = {}
        self.desired_state = self.params.get('state', None)
        self.verbose = module.params.get('verbose', False)

    def update_ntp_servers(self, host, ntp_servers_configured, ntp_servers_to_change, operation='overwrite'):
        """Update NTP server configuration"""
        host_date_time_manager = host.configManager.dateTimeSystem
        if host_date_time_manager:
            if operation == 'overwrite':
                new_ntp_servers = list(ntp_servers_to_change)
            else:
                new_ntp_servers = list(ntp_servers_configured)
                if operation == 'add':
                    new_ntp_servers = new_ntp_servers + ntp_servers_to_change
                elif operation == 'delete':
                    for server in ntp_servers_to_change:
                        if server in new_ntp_servers:
                            new_ntp_servers.remove(server)
            if self.verbose:
                message = self.build_changed_message(ntp_servers_configured, new_ntp_servers, ntp_servers_to_change, operation)
            ntp_config_spec = vim.host.NtpConfig()
            ntp_config_spec.server = new_ntp_servers
            date_config_spec = vim.host.DateTimeConfig()
            date_config_spec.ntpConfig = ntp_config_spec
            try:
                if not self.module.check_mode:
                    host_date_time_manager.UpdateDateTimeConfig(date_config_spec)
                if self.verbose:
                    self.results[host.name]['msg'] = message
            except vim.fault.HostConfigFault as config_fault:
                self.module.fail_json(msg="Failed to configure NTP for host '%s' due to : %s" % (host.name, to_native(config_fault.msg)))
            return new_ntp_servers

    def check_host_state(self):
        """Check ESXi host configuration"""
        change_list = []
        changed = False
        for host in self.hosts:
            self.results[host.name] = dict()
            if host.runtime.connectionState == 'connected':
                ntp_servers_configured, ntp_servers_to_change = self.check_ntp_servers(host=host)
                if self.desired_state:
                    self.results[host.name]['state'] = self.desired_state
                    if ntp_servers_to_change:
                        self.results[host.name]['ntp_servers_changed'] = ntp_servers_to_change
                        operation = 'add' if self.desired_state == 'present' else 'delete'
                        new_ntp_servers = self.update_ntp_servers(host=host, ntp_servers_configured=ntp_servers_configured, ntp_servers_to_change=ntp_servers_to_change, operation=operation)
                        self.results[host.name]['ntp_servers_current'] = new_ntp_servers
                        self.results[host.name]['changed'] = True
                        change_list.append(True)
                    else:
                        self.results[host.name]['ntp_servers_current'] = ntp_servers_configured
                        if self.verbose:
                            self.results[host.name]['msg'] = 'NTP servers already added' if self.desired_state == 'present' else 'NTP servers already removed'
                        self.results[host.name]['changed'] = False
                        change_list.append(False)
                else:
                    self.results[host.name]['ntp_servers'] = self.ntp_servers
                    if ntp_servers_to_change:
                        self.results[host.name]['ntp_servers_changed'] = self.get_differt_entries(ntp_servers_configured, ntp_servers_to_change)
                        self.update_ntp_servers(host=host, ntp_servers_configured=ntp_servers_configured, ntp_servers_to_change=ntp_servers_to_change, operation='overwrite')
                        self.results[host.name]['changed'] = True
                        change_list.append(True)
                    else:
                        if self.verbose:
                            self.results[host.name]['msg'] = 'NTP servers already configured'
                        self.results[host.name]['changed'] = False
                        change_list.append(False)
            else:
                self.results[host.name]['changed'] = False
                self.results[host.name]['msg'] = 'Host %s is disconnected and cannot be changed.' % host.name
        if any(change_list):
            changed = True
        self.module.exit_json(changed=changed, host_ntp_status=self.results)

    def check_ntp_servers(self, host):
        """Check configured NTP servers"""
        update_ntp_list = []
        host_datetime_system = host.configManager.dateTimeSystem
        if host_datetime_system:
            ntp_servers_configured = host_datetime_system.dateTimeInfo.ntpConfig.server
            if self.desired_state:
                for ntp_server in self.ntp_servers:
                    if self.desired_state == 'present' and ntp_server not in ntp_servers_configured:
                        update_ntp_list.append(ntp_server)
                    if self.desired_state == 'absent' and ntp_server in ntp_servers_configured:
                        update_ntp_list.append(ntp_server)
            elif ntp_servers_configured != self.ntp_servers:
                for ntp_server in self.ntp_servers:
                    update_ntp_list.append(ntp_server)
            if update_ntp_list:
                self.results[host.name]['ntp_servers_previous'] = ntp_servers_configured
        return (ntp_servers_configured, update_ntp_list)

    def build_changed_message(self, ntp_servers_configured, new_ntp_servers, ntp_servers_to_change, operation):
        """Build changed message"""
        check_mode = 'would be ' if self.module.check_mode else ''
        if operation == 'overwrite':
            add = self.get_not_in_list_one(new_ntp_servers, ntp_servers_configured)
            remove = self.get_not_in_list_one(ntp_servers_configured, new_ntp_servers)
            diff_servers = list(ntp_servers_configured)
            if add and remove:
                for server in add:
                    diff_servers.append(server)
                for server in remove:
                    diff_servers.remove(server)
                if new_ntp_servers != diff_servers:
                    message = 'NTP server %s %sadded and %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, self.array_to_string(remove), check_mode, check_mode)
                elif new_ntp_servers != ntp_servers_configured:
                    message = 'NTP server %s %sreplaced with %s' % (self.array_to_string(remove), check_mode, self.array_to_string(add))
                else:
                    message = 'NTP server %s %sremoved and %s %sadded' % (self.array_to_string(remove), check_mode, self.array_to_string(add), check_mode)
            elif add:
                for server in add:
                    diff_servers.append(server)
                if new_ntp_servers != diff_servers:
                    message = 'NTP server %s %sadded and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, check_mode)
                else:
                    message = 'NTP server %s %sadded' % (self.array_to_string(add), check_mode)
            elif remove:
                for server in remove:
                    diff_servers.remove(server)
                if new_ntp_servers != diff_servers:
                    message = 'NTP server %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(remove), check_mode, check_mode)
                else:
                    message = 'NTP server %s %sremoved' % (self.array_to_string(remove), check_mode)
            else:
                message = 'NTP server sequence %schanged' % check_mode
        elif operation == 'add':
            message = 'NTP server %s %sadded' % (self.array_to_string(ntp_servers_to_change), check_mode)
        elif operation == 'delete':
            message = 'NTP server %s %sremoved' % (self.array_to_string(ntp_servers_to_change), check_mode)
        return message

    @staticmethod
    def get_not_in_list_one(list1, list2):
        """Return entries that ore not in list one"""
        return [x for x in list1 if x not in set(list2)]

    @staticmethod
    def array_to_string(array):
        """Return string from array"""
        if len(array) > 2:
            string = ', '.join(("'{0}'".format(element) for element in array[:-1])) + ', and ' + "'{0}'".format(str(array[-1]))
        elif len(array) == 2:
            string = ' and '.join(("'{0}'".format(element) for element in array))
        elif len(array) == 1:
            string = "'{0}'".format(array[0])
        return string

    @staticmethod
    def get_differt_entries(list1, list2):
        """Return different entries of two lists"""
        return [a for a in list1 + list2 if a not in list1 or a not in list2]