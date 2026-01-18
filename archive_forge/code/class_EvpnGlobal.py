from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
class EvpnGlobal(object):
    """Manage global configuration of EVPN"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.overlay_enable = self.module.params['evpn_overlay_enable']
        self.commands = list()
        self.global_info = dict()
        self.conf_exist = False
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def init_module(self):
        """init_module"""
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

    def get_evpn_global_info(self):
        """ get current EVPN global configuration"""
        self.global_info['evpnOverLay'] = 'disable'
        cmd = 'display current-configuration | include ^evpn-overlay enable'
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        if out:
            self.global_info['evpnOverLay'] = 'enable'

    def get_existing(self):
        """get existing config"""
        self.existing = dict(evpn_overlay_enable=self.global_info['evpnOverLay'])

    def get_proposed(self):
        """get proposed config"""
        self.proposed = dict(evpn_overlay_enable=self.overlay_enable)

    def get_end_state(self):
        """get end config"""
        self.get_evpn_global_info()
        self.end_state = dict(evpn_overlay_enable=self.global_info['evpnOverLay'])

    def show_result(self):
        """ show result"""
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)

    def judge_if_config_exist(self):
        """ judge whether configuration has existed"""
        if self.overlay_enable == self.global_info['evpnOverLay']:
            return True
        return False

    def config_evnp_global(self):
        """ set global EVPN configuration"""
        if not self.conf_exist:
            if self.overlay_enable == 'enable':
                self.cli_add_command('evpn-overlay enable')
            else:
                self.cli_add_command('evpn-overlay enable', True)
            if self.commands:
                self.cli_load_config(self.commands)
                self.changed = True

    def work(self):
        """execute task"""
        self.get_evpn_global_info()
        self.get_existing()
        self.get_proposed()
        self.conf_exist = self.judge_if_config_exist()
        self.config_evnp_global()
        self.get_end_state()
        self.show_result()