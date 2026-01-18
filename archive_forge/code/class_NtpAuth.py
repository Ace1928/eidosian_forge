from __future__ import (absolute_import, division, print_function)
import copy
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
class NtpAuth(object):
    """Manage ntp authentication"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.key_id = self.module.params['key_id']
        self.password = self.module.params['auth_pwd'] or None
        self.auth_mode = self.module.params['auth_mode'] or None
        self.auth_type = self.module.params['auth_type']
        self.trusted_key = self.module.params['trusted_key']
        self.authentication = self.module.params['authentication'] or None
        self.state = self.module.params['state']
        self.check_params()
        self.ntp_auth_conf = dict()
        self.key_id_exist = False
        self.cur_trusted_key = 'disable'
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = list()
        self.end_state = list()
        self.get_ntp_auth_exist_config()

    def check_params(self):
        """Check all input params"""
        if not self.key_id.isdigit():
            self.module.fail_json(msg='Error: key_id is not digit.')
        if int(self.key_id) < 1 or int(self.key_id) > 4294967295:
            self.module.fail_json(msg='Error: The length of key_id is between 1 and 4294967295.')
        if self.state == 'present' and (not self.password):
            self.module.fail_json(msg='Error: The password cannot be empty.')
        if self.state == 'present' and self.password:
            if self.auth_type == 'encrypt' and (len(self.password) < 20 or len(self.password) > 392):
                self.module.fail_json(msg='Error: The length of encrypted password is between 20 and 392.')
            elif self.auth_type == 'text' and (len(self.password) < 1 or len(self.password) > 255):
                self.module.fail_json(msg='Error: The length of text password is between 1 and 255.')

    def init_module(self):
        """Init module object"""
        required_if = [('state', 'present', ('auth_pwd', 'auth_mode'))]
        self.module = AnsibleModule(argument_spec=self.spec, required_if=required_if, supports_check_mode=True)

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

    def get_ntp_auth_enable(self):
        """Get ntp authentication enable state"""
        flags = list()
        exp = '| exclude undo | include ntp authentication'
        flags.append(exp)
        config = self.get_config(flags)
        auth_en = re.findall('.*ntp\\s*authentication\\s*enable.*', config)
        if auth_en:
            self.ntp_auth_conf['authentication'] = 'enable'
        else:
            self.ntp_auth_conf['authentication'] = 'disable'

    def get_ntp_all_auth_keyid(self):
        """Get all authentication keyid info"""
        ntp_auth_conf = list()
        flags = list()
        exp = '| include authentication-keyid %s' % self.key_id
        flags.append(exp)
        config = self.get_config(flags)
        ntp_config_list = config.split('\n')
        if not ntp_config_list:
            self.ntp_auth_conf['authentication-keyid'] = 'None'
            return ntp_auth_conf
        self.key_id_exist = True
        cur_auth_mode = ''
        cur_auth_pwd = ''
        for ntp_config in ntp_config_list:
            ntp_auth_mode = re.findall('.*authentication-mode(\\s\\S*)\\s\\S*\\s(\\S*)', ntp_config)
            ntp_auth_trust = re.findall('.*trusted.*', ntp_config)
            if ntp_auth_trust:
                self.cur_trusted_key = 'enable'
            if ntp_auth_mode:
                cur_auth_mode = ntp_auth_mode[0][0].strip()
                cur_auth_pwd = ntp_auth_mode[0][1].strip()
        ntp_auth_conf.append(dict(key_id=self.key_id, auth_mode=cur_auth_mode, auth_pwd=cur_auth_pwd, trusted_key=self.cur_trusted_key))
        self.ntp_auth_conf['authentication-keyid'] = ntp_auth_conf
        return ntp_auth_conf

    def get_ntp_auth_exist_config(self):
        """Get ntp authentication existed configure"""
        self.get_ntp_auth_enable()
        self.get_ntp_all_auth_keyid()

    def config_ntp_auth_keyid(self):
        """Config ntp authentication keyid"""
        commands = list()
        if self.auth_type == 'encrypt':
            config_cli = 'ntp authentication-keyid %s authentication-mode %s cipher %s' % (self.key_id, self.auth_mode, self.password)
        else:
            config_cli = 'ntp authentication-keyid %s authentication-mode %s %s' % (self.key_id, self.auth_mode, self.password)
        commands.append(config_cli)
        if self.trusted_key != self.cur_trusted_key:
            if self.trusted_key == 'enable':
                config_cli_trust = 'ntp trusted authentication-keyid %s' % self.key_id
                commands.append(config_cli_trust)
            else:
                config_cli_trust = 'undo ntp trusted authentication-keyid %s' % self.key_id
                commands.append(config_cli_trust)
        self.cli_load_config(commands)

    def config_ntp_auth_enable(self):
        """Config ntp authentication enable"""
        commands = list()
        if self.ntp_auth_conf['authentication'] != self.authentication:
            if self.authentication == 'enable':
                config_cli = 'ntp authentication enable'
            else:
                config_cli = 'undo ntp authentication enable'
            commands.append(config_cli)
            self.cli_load_config(commands)

    def undo_config_ntp_auth_keyid(self):
        """Undo ntp authentication key-id"""
        commands = list()
        config_cli = 'undo ntp authentication-keyid %s' % self.key_id
        commands.append(config_cli)
        self.cli_load_config(commands)

    def cli_load_config(self, commands):
        """Load config by cli"""
        if not self.module.check_mode:
            load_config(self.module, commands)

    def config_ntp_auth(self):
        """Config ntp authentication"""
        if self.state == 'present':
            self.config_ntp_auth_keyid()
        else:
            if not self.key_id_exist:
                self.module.fail_json(msg='Error: The Authentication-keyid does not exist.')
            self.undo_config_ntp_auth_keyid()
        if self.authentication:
            self.config_ntp_auth_enable()
        self.changed = True

    def get_existing(self):
        """Get existing info"""
        self.existing = copy.deepcopy(self.ntp_auth_conf)

    def get_proposed(self):
        """Get proposed result"""
        auth_type = self.auth_type
        trusted_key = self.trusted_key
        if self.state == 'absent':
            auth_type = None
            trusted_key = None
        self.proposed = dict(key_id=self.key_id, auth_pwd=self.password, auth_mode=self.auth_mode, auth_type=auth_type, trusted_key=trusted_key, authentication=self.authentication, state=self.state)

    def get_update_cmd(self):
        """Get updated commands"""
        cli_str = ''
        if self.state == 'present':
            cli_str = 'ntp authentication-keyid %s authentication-mode %s ' % (self.key_id, self.auth_mode)
            if self.auth_type == 'encrypt':
                cli_str = '%s cipher %s' % (cli_str, self.password)
            else:
                cli_str = '%s %s' % (cli_str, self.password)
        else:
            cli_str = 'undo ntp authentication-keyid %s' % self.key_id
        self.updates_cmd.append(cli_str)
        if self.authentication:
            cli_str = ''
            if self.ntp_auth_conf['authentication'] != self.authentication:
                if self.authentication == 'enable':
                    cli_str = 'ntp authentication enable'
                else:
                    cli_str = 'undo ntp authentication enable'
            if cli_str != '':
                self.updates_cmd.append(cli_str)
        cli_str = ''
        if self.state == 'present':
            if self.trusted_key != self.cur_trusted_key:
                if self.trusted_key == 'enable':
                    cli_str = 'ntp trusted authentication-keyid %s' % self.key_id
                else:
                    cli_str = 'undo ntp trusted authentication-keyid %s' % self.key_id
        else:
            cli_str = 'undo ntp trusted authentication-keyid %s' % self.key_id
        if cli_str != '':
            self.updates_cmd.append(cli_str)

    def get_end_state(self):
        """Get end state info"""
        self.ntp_auth_conf = dict()
        self.get_ntp_auth_exist_config()
        self.end_state = copy.deepcopy(self.ntp_auth_conf)
        if self.end_state == self.existing:
            self.changed = False

    def show_result(self):
        """Show result"""
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)

    def work(self):
        """Execute task"""
        self.get_existing()
        self.get_proposed()
        self.get_update_cmd()
        self.config_ntp_auth()
        self.get_end_state()
        self.show_result()