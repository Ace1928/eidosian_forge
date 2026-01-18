from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, exec_command, run_commands
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList
class RollBack(object):
    """
    Manages rolls back the system from the current configuration state to a historical configuration state.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)
        self.commands = list()
        self.commit_id = self.module.params['commit_id']
        self.label = self.module.params['label']
        self.filename = self.module.params['filename']
        self.last = self.module.params['last']
        self.oldest = self.module.params['oldest']
        self.action = self.module.params['action']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.existing = dict()
        self.proposed = dict()
        self.end_state = dict()
        self.rollback_info = None
        self.init_module()

    def init_module(self):
        """ init module """
        required_if = [('action', 'set', ['commit_id', 'label']), ('action', 'commit', ['label'])]
        mutually_exclusive = None
        required_one_of = None
        if self.action == 'rollback':
            required_one_of = [['commit_id', 'label', 'filename', 'last']]
        elif self.action == 'clear':
            required_one_of = [['commit_id', 'oldest']]
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True, required_if=required_if, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        self.commands.append('return')
        self.commands.append('mmi-mode enable')
        if self.action == 'commit':
            self.commands.append('sys')
        self.commands.append(command)
        self.updates_cmd.append(command)

    def cli_load_config(self, commands):
        """load config by cli"""
        if not self.module.check_mode:
            run_commands(self.module, commands)

    def get_config(self, flags=None):
        """Retrieves the current config from the device or cache
        """
        flags = [] if flags is None else flags
        cmd = 'display configuration '
        cmd += ' '.join(flags)
        cmd = cmd.strip()
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        cfg = str(out).strip()
        return cfg

    def get_rollback_dict(self):
        """ get rollback attributes dict."""
        rollback_info = dict()
        rollback_info['RollBackInfos'] = list()
        flags = list()
        exp = 'commit list'
        flags.append(exp)
        cfg_info = self.get_config(flags)
        if not cfg_info:
            return rollback_info
        cfg_line = cfg_info.split('\n')
        for cfg in cfg_line:
            if re.findall('^\\d', cfg):
                pre_rollback_info = cfg.split()
                rollback_info['RollBackInfos'].append(dict(commitId=pre_rollback_info[1].replace('*', ''), userLabel=pre_rollback_info[2]))
        return rollback_info

    def get_filename_type(self, filename):
        """Gets the type of filename, such as cfg, zip, dat..."""
        if filename is None:
            return None
        if ' ' in filename:
            self.module.fail_json(msg='Error: Configuration file name include spaces.')
        iftype = None
        if filename.endswith('.cfg'):
            iftype = 'cfg'
        elif filename.endswith('.zip'):
            iftype = 'zip'
        elif filename.endswith('.dat'):
            iftype = 'dat'
        else:
            return None
        return iftype.lower()

    def set_config(self):
        if self.action == 'rollback':
            if self.commit_id:
                cmd = 'rollback configuration to commit-id %s' % self.commit_id
                self.cli_add_command(cmd)
            if self.label:
                cmd = 'rollback configuration to label %s' % self.label
                self.cli_add_command(cmd)
            if self.filename:
                cmd = 'rollback configuration to file %s' % self.filename
                self.cli_add_command(cmd)
            if self.last:
                cmd = 'rollback configuration last %s' % self.last
                self.cli_add_command(cmd)
        elif self.action == 'set':
            if self.commit_id and self.label:
                cmd = 'set configuration commit %s label %s' % (self.commit_id, self.label)
                self.cli_add_command(cmd)
        elif self.action == 'clear':
            if self.commit_id:
                cmd = 'clear configuration commit %s label' % self.commit_id
                self.cli_add_command(cmd)
            if self.oldest:
                cmd = 'clear configuration commit oldest %s' % self.oldest
                self.cli_add_command(cmd)
        elif self.action == 'commit':
            if self.label:
                cmd = 'commit label %s' % self.label
                self.cli_add_command(cmd)
        elif self.action == 'display':
            self.rollback_info = self.get_rollback_dict()
        if self.commands:
            self.commands.append('return')
            self.commands.append('undo mmi-mode enable')
            self.cli_load_config(self.commands)
            self.changed = True

    def check_params(self):
        """Check all input params"""
        rollback_info = self.rollback_info['RollBackInfos']
        if self.commit_id:
            if not self.commit_id.isdigit():
                self.module.fail_json(msg='Error: The parameter of commit_id is invalid.')
            info_bool = False
            for info in rollback_info:
                if info.get('commitId') == self.commit_id:
                    info_bool = True
            if not info_bool:
                self.module.fail_json(msg='Error: The parameter of commit_id is not exist.')
            if self.action == 'clear':
                info_bool = False
                for info in rollback_info:
                    if info.get('commitId') == self.commit_id:
                        if info.get('userLabel') == '-':
                            info_bool = True
                if info_bool:
                    self.module.fail_json(msg='Error: This commit_id does not have a label.')
        if self.filename:
            if not self.get_filename_type(self.filename):
                self.module.fail_json(msg='Error: Invalid file name or file name extension ( *.cfg, *.zip, *.dat ).')
        if self.last:
            if not self.last.isdigit():
                self.module.fail_json(msg='Error: Number of configuration checkpoints is not digit.')
            if int(self.last) <= 0 or int(self.last) > 80:
                self.module.fail_json(msg='Error: Number of configuration checkpoints is not in the range from 1 to 80.')
        if self.oldest:
            if not self.oldest.isdigit():
                self.module.fail_json(msg='Error: Number of configuration checkpoints is not digit.')
            if int(self.oldest) <= 0 or int(self.oldest) > 80:
                self.module.fail_json(msg='Error: Number of configuration checkpoints is not in the range from 1 to 80.')
        if self.label:
            if self.label[0].isdigit():
                self.module.fail_json(msg='Error: Commit label which should not start with a number.')
            if len(self.label.replace(' ', '')) == 1:
                if self.label == '-':
                    self.module.fail_json(msg='Error: Commit label which should not be "-"')
            if len(self.label.replace(' ', '')) < 1 or len(self.label) > 256:
                self.module.fail_json(msg='Error: Label of configuration checkpoints is a string of 1 to 256 characters.')
            if self.action == 'rollback':
                info_bool = False
                for info in rollback_info:
                    if info.get('userLabel') == self.label:
                        info_bool = True
                if not info_bool:
                    self.module.fail_json(msg='Error: The parameter of userLabel is not exist.')
            if self.action == 'commit':
                info_bool = False
                for info in rollback_info:
                    if info.get('userLabel') == self.label:
                        info_bool = True
                if info_bool:
                    self.module.fail_json(msg='Error: The parameter of userLabel is existing.')
            if self.action == 'set':
                info_bool = False
                for info in rollback_info:
                    if info.get('commitId') == self.commit_id:
                        if info.get('userLabel') != '-':
                            info_bool = True
                if info_bool:
                    self.module.fail_json(msg='Error: The userLabel of this commitid is present and can be reset after deletion.')

    def get_proposed(self):
        """get proposed info"""
        if self.commit_id:
            self.proposed['commit_id'] = self.commit_id
        if self.label:
            self.proposed['label'] = self.label
        if self.filename:
            self.proposed['filename'] = self.filename
        if self.last:
            self.proposed['last'] = self.last
        if self.oldest:
            self.proposed['oldest'] = self.oldest

    def get_existing(self):
        """get existing info"""
        if not self.rollback_info:
            self.existing['RollBackInfos'] = None
        else:
            self.existing['RollBackInfos'] = self.rollback_info['RollBackInfos']

    def get_end_state(self):
        """get end state info"""
        rollback_info = self.get_rollback_dict()
        if not rollback_info:
            self.end_state['RollBackInfos'] = None
        else:
            self.end_state['RollBackInfos'] = rollback_info['RollBackInfos']

    def work(self):
        """worker"""
        self.rollback_info = self.get_rollback_dict()
        self.check_params()
        self.get_proposed()
        self.set_config()
        self.get_existing()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)