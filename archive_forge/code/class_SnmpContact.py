from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
class SnmpContact(object):
    """ Manages SNMP contact configuration """

    def __init__(self, **kwargs):
        """ Class init """
        argument_spec = kwargs['argument_spec']
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)
        self.cur_cfg = dict()
        self.state = self.module.params['state']
        self.contact = self.module.params['contact']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def check_args(self):
        """ Check invalid args """
        if self.contact:
            if len(self.contact) > 255 or len(self.contact) < 1:
                self.module.fail_json(msg='Error: The len of contact %s is out of [1 - 255].' % self.contact)
        else:
            self.module.fail_json(msg='Error: The len of contact is 0.')

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

    def get_proposed(self):
        """ Get proposed state """
        self.proposed['state'] = self.state
        if self.contact:
            self.proposed['contact'] = self.contact

    def get_existing(self):
        """ Get existing state """
        tmp_cfg = self.cli_get_config()
        if tmp_cfg:
            temp_data = tmp_cfg.split('contact ')
            if len(temp_data) > 1:
                self.cur_cfg['contact'] = temp_data[1]
                self.existing['contact'] = temp_data[1]

    def get_end_state(self):
        """ Get end state """
        tmp_cfg = self.cli_get_config()
        if tmp_cfg:
            temp_data = tmp_cfg.split('contact ')
            if len(temp_data) > 1:
                self.end_state['contact'] = temp_data[1]

    def cli_load_config(self, commands):
        """ Load configure by cli """
        if not self.module.check_mode:
            load_config(self.module, commands)

    def cli_get_config(self):
        """ Get configure by cli """
        regular = '| include snmp | include contact'
        flags = list()
        flags.append(regular)
        tmp_cfg = self.get_config(flags)
        return tmp_cfg

    def set_config(self):
        """ Set configure by cli """
        cmd = 'snmp-agent sys-info contact %s' % self.contact
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def undo_config(self):
        """ Undo configure by cli """
        cmd = 'undo snmp-agent sys-info contact'
        self.updates_cmd.append(cmd)
        cmds = list()
        cmds.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True

    def work(self):
        """ Main work function """
        self.check_args()
        self.get_proposed()
        self.get_existing()
        if self.state == 'present':
            if 'contact' in self.cur_cfg.keys() and self.contact == self.cur_cfg['contact']:
                pass
            else:
                self.set_config()
        elif 'contact' in self.cur_cfg.keys() and self.contact == self.cur_cfg['contact']:
            self.undo_config()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        self.results['updates'] = self.updates_cmd
        self.module.exit_json(**self.results)