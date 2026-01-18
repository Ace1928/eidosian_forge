from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_config, exec_command, cli_err_msg
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
class AclInterface(object):
    """ Manages acl interface configuration """

    def __init__(self, **kwargs):
        """ Class init """
        argument_spec = kwargs['argument_spec']
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)
        self.cur_cfg = dict()
        self.cur_cfg['acl interface'] = []
        self.state = self.module.params['state']
        self.acl_name = self.module.params['acl_name']
        self.interface = self.module.params['interface']
        self.direction = self.module.params['direction']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def check_args(self):
        """ Check args """
        if self.acl_name:
            if self.acl_name.isdigit():
                if int(self.acl_name) < 2000 or int(self.acl_name) > 4999:
                    self.module.fail_json(msg='Error: The value of acl_name is out of [2000 - 4999].')
            elif len(self.acl_name) < 1 or len(self.acl_name) > 32:
                self.module.fail_json(msg='Error: The len of acl_name is out of [1 - 32].')
        if self.interface:
            cmd = 'display current-configuration | ignore-case section include interface %s' % self.interface
            rc, out, err = exec_command(self.module, cmd)
            if rc != 0:
                self.module.fail_json(msg=err)
            result = str(out).strip()
            if result:
                tmp = result.split('\n')
                if 'display' in tmp[0]:
                    tmp.pop(0)
                if not tmp:
                    self.module.fail_json(msg='Error: The interface %s is not in the device.' % self.interface)

    def get_proposed(self):
        """ Get proposed config """
        self.proposed['state'] = self.state
        if self.acl_name:
            self.proposed['acl_name'] = self.acl_name
        if self.interface:
            self.proposed['interface'] = self.interface
        if self.direction:
            self.proposed['direction'] = self.direction

    def get_existing(self):
        """ Get existing config """
        cmd = 'display current-configuration | ignore-case section include interface %s | include traffic-filter' % self.interface
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        result = str(out).strip()
        end = []
        if result:
            tmp = result.split('\n')
            if 'display' in tmp[0]:
                tmp.pop(0)
            for item in tmp:
                end.append(item.strip())
            self.cur_cfg['acl interface'] = end
            self.existing['acl interface'] = end

    def get_end_state(self):
        """ Get config end state """
        cmd = 'display current-configuration | ignore-case section include interface %s | include traffic-filter' % self.interface
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        result = str(out).strip()
        end = []
        if result:
            tmp = result.split('\n')
            if 'display' in tmp[0]:
                tmp.pop(0)
            for item in tmp:
                end.append(item.strip())
            self.end_state['acl interface'] = end

    def load_config(self, config):
        """Sends configuration commands to the remote device"""
        rc, out, err = exec_command(self.module, 'mmi-mode enable')
        if rc != 0:
            self.module.fail_json(msg='unable to set mmi-mode enable', output=err)
        rc, out, err = exec_command(self.module, 'system-view immediately')
        if rc != 0:
            self.module.fail_json(msg='unable to enter system-view', output=err)
        for cmd in config:
            rc, out, err = exec_command(self.module, cmd)
            if rc != 0:
                if 'unrecognized command found' in err.lower():
                    self.module.fail_json(msg='Error:The parameter is incorrect or the interface does not support this parameter.')
                else:
                    self.module.fail_json(msg=cli_err_msg(cmd.strip(), err))
        exec_command(self.module, 'return')

    def cli_load_config(self, commands):
        """ Cli method to load config """
        if not self.module.check_mode:
            self.load_config(commands)

    def work(self):
        """ Work function """
        self.check_args()
        self.get_proposed()
        self.get_existing()
        cmds = list()
        tmp_cmd = 'traffic-filter acl %s %s' % (self.acl_name, self.direction)
        undo_tmp_cmd = 'undo traffic-filter acl %s %s' % (self.acl_name, self.direction)
        if self.state == 'present':
            if tmp_cmd not in self.cur_cfg['acl interface']:
                interface_cmd = 'interface %s' % self.interface.lower()
                cmds.append(interface_cmd)
                cmds.append(tmp_cmd)
                self.cli_load_config(cmds)
                self.changed = True
                self.updates_cmd.append(interface_cmd)
                self.updates_cmd.append(tmp_cmd)
        elif tmp_cmd in self.cur_cfg['acl interface']:
            interface_cmd = 'interface %s' % self.interface
            cmds.append(interface_cmd)
            cmds.append(undo_tmp_cmd)
            self.cli_load_config(cmds)
            self.changed = True
            self.updates_cmd.append(interface_cmd)
            self.updates_cmd.append(undo_tmp_cmd)
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        self.results['updates'] = self.updates_cmd
        self.module.exit_json(**self.results)