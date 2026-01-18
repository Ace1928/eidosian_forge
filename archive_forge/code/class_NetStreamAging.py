from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
class NetStreamAging(object):
    """
    Manages netstream aging.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.timeout_interval = self.module.params['timeout_interval']
        self.type = self.module.params['type']
        self.state = self.module.params['state']
        self.timeout_type = self.module.params['timeout_type']
        self.manual_slot = self.module.params['manual_slot']
        self.host = self.module.params['host']
        self.username = self.module.params['username']
        self.port = self.module.params['port']
        self.changed = False
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.existing['active_timeout'] = list()
        self.existing['inactive_timeout'] = list()
        self.existing['tcp_timeout'] = list()
        self.end_state['active_timeout'] = list()
        self.end_state['inactive_timeout'] = list()
        self.end_state['tcp_timeout'] = list()
        self.active_changed = False
        self.inactive_changed = False
        self.tcp_changed = False

    def init_module(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def cli_load_config(self, commands):
        """load config by cli"""
        if not self.module.check_mode:
            load_config(self.module, commands)

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        if undo and command.lower() not in ['quit', 'return']:
            cmd = 'undo ' + command
        else:
            cmd = command
        self.commands.append(cmd)
        if command.lower() not in ['quit', 'return']:
            self.updates_cmd.append(cmd)

    def get_exist_timer_out_para(self):
        """Get exist netstream timeout parameters"""
        active_tmp = dict()
        inactive_tmp = dict()
        tcp_tmp = dict()
        active_tmp['ip'] = '30'
        active_tmp['vxlan'] = '30'
        inactive_tmp['ip'] = '30'
        inactive_tmp['vxlan'] = '30'
        tcp_tmp['ip'] = 'absent'
        tcp_tmp['vxlan'] = 'absent'
        cmd = 'display current-configuration | include ^netstream timeout'
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        config = str(out).strip()
        if config:
            config = config.lstrip()
            config_list = config.split('\n')
            for config_mem in config_list:
                config_mem = config_mem.lstrip()
                config_mem_list = config_mem.split(' ')
                if len(config_mem_list) > 4 and config_mem_list[2] == 'ip':
                    if config_mem_list[3] == 'active':
                        active_tmp['ip'] = config_mem_list[4]
                    if config_mem_list[3] == 'inactive':
                        inactive_tmp['ip'] = config_mem_list[4]
                    if config_mem_list[3] == 'tcp-session':
                        tcp_tmp['ip'] = 'present'
                if len(config_mem_list) > 4 and config_mem_list[2] == 'vxlan':
                    if config_mem_list[4] == 'active':
                        active_tmp['vxlan'] = config_mem_list[5]
                    if config_mem_list[4] == 'inactive':
                        inactive_tmp['vxlan'] = config_mem_list[5]
                    if config_mem_list[4] == 'tcp-session':
                        tcp_tmp['vxlan'] = 'present'
        self.existing['active_timeout'].append(active_tmp)
        self.existing['inactive_timeout'].append(inactive_tmp)
        self.existing['tcp_timeout'].append(tcp_tmp)

    def get_end_timer_out_para(self):
        """Get end netstream timeout parameters"""
        active_tmp = dict()
        inactive_tmp = dict()
        tcp_tmp = dict()
        active_tmp['ip'] = '30'
        active_tmp['vxlan'] = '30'
        inactive_tmp['ip'] = '30'
        inactive_tmp['vxlan'] = '30'
        tcp_tmp['ip'] = 'absent'
        tcp_tmp['vxlan'] = 'absent'
        cmd = 'display current-configuration | include ^netstream timeout'
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        config = str(out).strip()
        if config:
            config = config.lstrip()
            config_list = config.split('\n')
            for config_mem in config_list:
                config_mem = config_mem.lstrip()
                config_mem_list = config_mem.split(' ')
                if len(config_mem_list) > 4 and config_mem_list[2] == 'ip':
                    if config_mem_list[3] == 'active':
                        active_tmp['ip'] = config_mem_list[4]
                    if config_mem_list[3] == 'inactive':
                        inactive_tmp['ip'] = config_mem_list[4]
                    if config_mem_list[3] == 'tcp-session':
                        tcp_tmp['ip'] = 'present'
                if len(config_mem_list) > 4 and config_mem_list[2] == 'vxlan':
                    if config_mem_list[4] == 'active':
                        active_tmp['vxlan'] = config_mem_list[5]
                    if config_mem_list[4] == 'inactive':
                        inactive_tmp['vxlan'] = config_mem_list[5]
                    if config_mem_list[4] == 'tcp-session':
                        tcp_tmp['vxlan'] = 'present'
        self.end_state['active_timeout'].append(active_tmp)
        self.end_state['inactive_timeout'].append(inactive_tmp)
        self.end_state['tcp_timeout'].append(tcp_tmp)

    def check_params(self):
        """Check all input params"""
        if not str(self.timeout_interval).isdigit():
            self.module.fail_json(msg='Error: Timeout interval should be numerical.')
        if self.timeout_type == 'active':
            if int(self.timeout_interval) < 1 or int(self.timeout_interval) > 60:
                self.module.fail_json(msg='Error: Active interval should between 1 - 60 minutes.')
        if self.timeout_type == 'inactive':
            if int(self.timeout_interval) < 5 or int(self.timeout_interval) > 600:
                self.module.fail_json(msg='Error: Inactive interval should between 5 - 600 seconds.')
        if self.timeout_type == 'manual':
            if not self.manual_slot:
                self.module.fail_json(msg='Error: If use manual timeout mode,slot number is needed.')
            if re.match('^\\d+(\\/\\d*)?$', self.manual_slot) is None:
                self.module.fail_json(msg='Error: Slot number should be numerical.')

    def get_proposed(self):
        """get proposed info"""
        if self.timeout_interval:
            self.proposed['timeout_interval'] = self.timeout_interval
        if self.timeout_type:
            self.proposed['timeout_type'] = self.timeout_type
        if self.type:
            self.proposed['type'] = self.type
        if self.state:
            self.proposed['state'] = self.state
        if self.manual_slot:
            self.proposed['manual_slot'] = self.manual_slot

    def get_existing(self):
        """get existing info"""
        active_tmp = dict()
        inactive_tmp = dict()
        tcp_tmp = dict()
        self.get_exist_timer_out_para()
        if self.timeout_type == 'active':
            for active_tmp in self.existing['active_timeout']:
                if self.state == 'present':
                    if str(active_tmp[self.type]) != self.timeout_interval:
                        self.active_changed = True
                else:
                    if self.timeout_interval != '30':
                        if str(active_tmp[self.type]) != '30':
                            if str(active_tmp[self.type]) != self.timeout_interval:
                                self.module.fail_json(msg='Error: The specified active interval do not exist.')
                    if str(active_tmp[self.type]) != '30':
                        self.timeout_interval = active_tmp[self.type]
                        self.active_changed = True
        if self.timeout_type == 'inactive':
            for inactive_tmp in self.existing['inactive_timeout']:
                if self.state == 'present':
                    if str(inactive_tmp[self.type]) != self.timeout_interval:
                        self.inactive_changed = True
                else:
                    if self.timeout_interval != '30':
                        if str(inactive_tmp[self.type]) != '30':
                            if str(inactive_tmp[self.type]) != self.timeout_interval:
                                self.module.fail_json(msg='Error: The specified inactive interval do not exist.')
                    if str(inactive_tmp[self.type]) != '30':
                        self.timeout_interval = inactive_tmp[self.type]
                        self.inactive_changed = True
        if self.timeout_type == 'tcp-session':
            for tcp_tmp in self.existing['tcp_timeout']:
                if str(tcp_tmp[self.type]) != self.state:
                    self.tcp_changed = True

    def operate_time_out(self):
        """configure timeout parameters"""
        cmd = ''
        if self.timeout_type == 'manual':
            if self.type == 'ip':
                self.cli_add_command('quit')
                cmd = 'reset netstream cache ip slot %s' % self.manual_slot
                self.cli_add_command(cmd)
            elif self.type == 'vxlan':
                self.cli_add_command('quit')
                cmd = 'reset netstream cache vxlan inner-ip slot %s' % self.manual_slot
                self.cli_add_command(cmd)
        if not self.active_changed and (not self.inactive_changed) and (not self.tcp_changed):
            if self.commands:
                self.cli_load_config(self.commands)
                self.changed = True
            return
        if self.active_changed or self.inactive_changed:
            if self.type == 'ip':
                cmd = 'netstream timeout ip %s %s' % (self.timeout_type, self.timeout_interval)
            elif self.type == 'vxlan':
                cmd = 'netstream timeout vxlan inner-ip %s %s' % (self.timeout_type, self.timeout_interval)
            if self.state == 'absent':
                self.cli_add_command(cmd, undo=True)
            else:
                self.cli_add_command(cmd)
        if self.timeout_type == 'tcp-session' and self.tcp_changed:
            if self.type == 'ip':
                if self.state == 'present':
                    cmd = 'netstream timeout ip tcp-session'
                else:
                    cmd = 'undo netstream timeout ip tcp-session'
            elif self.type == 'vxlan':
                if self.state == 'present':
                    cmd = 'netstream timeout vxlan inner-ip tcp-session'
                else:
                    cmd = 'undo netstream timeout vxlan inner-ip tcp-session'
            self.cli_add_command(cmd)
        if self.commands:
            self.cli_load_config(self.commands)
            self.changed = True

    def get_end_state(self):
        """get end state info"""
        self.get_end_timer_out_para()

    def work(self):
        """worker"""
        self.check_params()
        self.get_existing()
        self.get_proposed()
        self.operate_time_out()
        self.get_end_state()
        if self.existing == self.end_state:
            self.changed = False
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)