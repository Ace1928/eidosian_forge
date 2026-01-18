from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config, ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
class SnmpTraps(object):
    """ Manages SNMP trap configuration """

    def __init__(self, **kwargs):
        """ Class init """
        argument_spec = kwargs['argument_spec']
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, required_together=[('interface_type', 'interface_number')], supports_check_mode=True)
        self.cur_cfg = dict()
        self.cur_cfg['snmp-agent trap'] = []
        self.cur_cfg['undo snmp-agent trap'] = []
        self.state = self.module.params['state']
        self.feature_name = self.module.params['feature_name']
        self.trap_name = self.module.params['trap_name']
        self.interface_type = self.module.params['interface_type']
        self.interface_number = self.module.params['interface_number']
        self.port_number = self.module.params['port_number']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.existing['snmp-agent trap'] = []
        self.existing['undo snmp-agent trap'] = []
        self.end_state = dict()
        self.end_state['snmp-agent trap'] = []
        self.end_state['undo snmp-agent trap'] = []
        commands = list()
        cmd1 = 'display interface brief'
        commands.append(cmd1)
        self.interface = run_commands(self.module, commands)

    def get_config(self, flags=None):
        """Retrieves the current config from the device or cache
        """
        flags = [] if flags is None else flags
        cmd = 'display current-configuration '
        cmd += ' '.join(flags)
        cmd = cmd.strip()
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        cfg = str(out).strip()
        return cfg

    def check_args(self):
        """ Check invalid args """
        if self.port_number:
            if self.port_number.isdigit():
                if int(self.port_number) < 1025 or int(self.port_number) > 65535:
                    self.module.fail_json(msg='Error: The value of port_number is out of [1025 - 65535].')
            else:
                self.module.fail_json(msg='Error: The port_number is not digit.')
        if self.interface_type and self.interface_number:
            tmp_interface = self.interface_type + self.interface_number
            if tmp_interface not in self.interface[0]:
                self.module.fail_json(msg='Error: The interface %s is not in the device.' % tmp_interface)

    def get_proposed(self):
        """ Get proposed state """
        self.proposed['state'] = self.state
        if self.feature_name:
            self.proposed['feature_name'] = self.feature_name
        if self.trap_name:
            self.proposed['trap_name'] = self.trap_name
        if self.interface_type:
            self.proposed['interface_type'] = self.interface_type
        if self.interface_number:
            self.proposed['interface_number'] = self.interface_number
        if self.port_number:
            self.proposed['port_number'] = self.port_number

    def get_existing(self):
        """ Get existing state """
        tmp_cfg = self.cli_get_config()
        if tmp_cfg:
            temp_cfg_lower = tmp_cfg.lower()
            temp_data = tmp_cfg.split('\n')
            temp_data_lower = temp_cfg_lower.split('\n')
            for item in temp_data:
                if 'snmp-agent trap source-port ' in item:
                    if self.port_number:
                        item_tmp = item.split('snmp-agent trap source-port ')
                        self.cur_cfg['trap source-port'] = item_tmp[1]
                        self.existing['trap source-port'] = item_tmp[1]
                elif 'snmp-agent trap source ' in item:
                    if self.interface_type:
                        item_tmp = item.split('snmp-agent trap source ')
                        self.cur_cfg['trap source interface'] = item_tmp[1]
                        self.existing['trap source interface'] = item_tmp[1]
            if self.feature_name:
                for item in temp_data_lower:
                    if item == 'snmp-agent trap enable':
                        self.cur_cfg['snmp-agent trap'].append('enable')
                        self.existing['snmp-agent trap'].append('enable')
                    elif item == 'snmp-agent trap disable':
                        self.cur_cfg['snmp-agent trap'].append('disable')
                        self.existing['snmp-agent trap'].append('disable')
                    elif 'undo snmp-agent trap enable ' in item:
                        item_tmp = item.split('undo snmp-agent trap enable ')
                        self.cur_cfg['undo snmp-agent trap'].append(item_tmp[1])
                        self.existing['undo snmp-agent trap'].append(item_tmp[1])
                    elif 'snmp-agent trap enable ' in item:
                        item_tmp = item.split('snmp-agent trap enable ')
                        self.cur_cfg['snmp-agent trap'].append(item_tmp[1])
                        self.existing['snmp-agent trap'].append(item_tmp[1])
            else:
                del self.existing['snmp-agent trap']
                del self.existing['undo snmp-agent trap']

    def get_end_state(self):
        """ Get end_state state """
        tmp_cfg = self.cli_get_config()
        if tmp_cfg:
            temp_cfg_lower = tmp_cfg.lower()
            temp_data = tmp_cfg.split('\n')
            temp_data_lower = temp_cfg_lower.split('\n')
            for item in temp_data:
                if 'snmp-agent trap source-port ' in item:
                    if self.port_number:
                        item_tmp = item.split('snmp-agent trap source-port ')
                        self.end_state['trap source-port'] = item_tmp[1]
                elif 'snmp-agent trap source ' in item:
                    if self.interface_type:
                        item_tmp = item.split('snmp-agent trap source ')
                        self.end_state['trap source interface'] = item_tmp[1]
            if self.feature_name:
                for item in temp_data_lower:
                    if item == 'snmp-agent trap enable':
                        self.end_state['snmp-agent trap'].append('enable')
                    elif item == 'snmp-agent trap disable':
                        self.end_state['snmp-agent trap'].append('disable')
                    elif 'undo snmp-agent trap enable ' in item:
                        item_tmp = item.split('undo snmp-agent trap enable ')
                        self.end_state['undo snmp-agent trap'].append(item_tmp[1])
                    elif 'snmp-agent trap enable ' in item:
                        item_tmp = item.split('snmp-agent trap enable ')
                        self.end_state['snmp-agent trap'].append(item_tmp[1])
            else:
                del self.end_state['snmp-agent trap']
                del self.end_state['undo snmp-agent trap']
        if self.end_state == self.existing:
            self.changed = False
            self.updates_cmd = list()

    def cli_load_config(self, commands):
        """ Load configure through cli """
        if not self.module.check_mode:
            load_config(self.module, commands)

    def cli_get_config(self):
        """ Get configure through cli """
        regular = '| include snmp | include trap'
        flags = list()
        flags.append(regular)
        tmp_cfg = self.get_config(flags)
        return tmp_cfg

    def set_trap_feature_name(self):
        """ Set feature name for trap """
        if self.feature_name == 'all':
            cmd = 'snmp-agent trap enable'
        else:
            cmd = 'snmp-agent trap enable feature-name %s' % self.feature_name
            if self.trap_name:
                cmd += ' trap-name %s' % self.trap_name
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def undo_trap_feature_name(self):
        """ Undo feature name for trap """
        if self.feature_name == 'all':
            cmd = 'undo snmp-agent trap enable'
        else:
            cmd = 'undo snmp-agent trap enable feature-name %s' % self.feature_name
            if self.trap_name:
                cmd += ' trap-name %s' % self.trap_name
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def set_trap_source_interface(self):
        """ Set source interface for trap """
        cmd = 'snmp-agent trap source %s %s' % (self.interface_type, self.interface_number)
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def undo_trap_source_interface(self):
        """ Undo source interface for trap """
        cmd = 'undo snmp-agent trap source'
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def set_trap_source_port(self):
        """ Set source port for trap """
        cmd = 'snmp-agent trap source-port %s' % self.port_number
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def undo_trap_source_port(self):
        """ Undo source port for trap """
        cmd = 'undo snmp-agent trap source-port'
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def work(self):
        """ The work function """
        self.check_args()
        self.get_proposed()
        self.get_existing()
        find_flag = False
        find_undo_flag = False
        tmp_interface = None
        if self.state == 'present':
            if self.feature_name:
                if self.trap_name:
                    tmp_cfg = 'feature-name %s trap-name %s' % (self.feature_name, self.trap_name.lower())
                else:
                    tmp_cfg = 'feature-name %s' % self.feature_name
                find_undo_flag = False
                if self.cur_cfg['undo snmp-agent trap']:
                    for item in self.cur_cfg['undo snmp-agent trap']:
                        if item == tmp_cfg:
                            find_undo_flag = True
                        elif tmp_cfg in item:
                            find_undo_flag = True
                        elif self.feature_name == 'all':
                            find_undo_flag = True
                if find_undo_flag:
                    self.set_trap_feature_name()
                if not find_undo_flag:
                    find_flag = False
                    if self.cur_cfg['snmp-agent trap']:
                        for item in self.cur_cfg['snmp-agent trap']:
                            if item == 'enable':
                                find_flag = True
                            elif item == tmp_cfg:
                                find_flag = True
                    if not find_flag:
                        self.set_trap_feature_name()
            if self.interface_type:
                find_flag = False
                tmp_interface = self.interface_type + self.interface_number
                if 'trap source interface' in self.cur_cfg.keys():
                    if self.cur_cfg['trap source interface'] == tmp_interface:
                        find_flag = True
                if not find_flag:
                    self.set_trap_source_interface()
            if self.port_number:
                find_flag = False
                if 'trap source-port' in self.cur_cfg.keys():
                    if self.cur_cfg['trap source-port'] == self.port_number:
                        find_flag = True
                if not find_flag:
                    self.set_trap_source_port()
        else:
            if self.feature_name:
                if self.trap_name:
                    tmp_cfg = 'feature-name %s trap-name %s' % (self.feature_name, self.trap_name.lower())
                else:
                    tmp_cfg = 'feature-name %s' % self.feature_name
                find_flag = False
                if self.cur_cfg['snmp-agent trap']:
                    for item in self.cur_cfg['snmp-agent trap']:
                        if item == tmp_cfg:
                            find_flag = True
                        elif item == 'enable':
                            find_flag = True
                        elif tmp_cfg in item:
                            find_flag = True
                else:
                    find_flag = True
                find_undo_flag = False
                if self.cur_cfg['undo snmp-agent trap']:
                    for item in self.cur_cfg['undo snmp-agent trap']:
                        if item == tmp_cfg:
                            find_undo_flag = True
                        elif tmp_cfg in item:
                            find_undo_flag = True
                if find_undo_flag:
                    pass
                elif find_flag:
                    self.undo_trap_feature_name()
            if self.interface_type:
                if 'trap source interface' in self.cur_cfg.keys():
                    self.undo_trap_source_interface()
            if self.port_number:
                if 'trap source-port' in self.cur_cfg.keys():
                    self.undo_trap_source_port()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        self.results['updates'] = self.updates_cmd
        self.module.exit_json(**self.results)