from __future__ import (absolute_import, division, print_function)
import sys
import socket
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
class BfdGlobal(object):
    """Manages BFD Global"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.bfd_enable = self.module.params['bfd_enable']
        self.default_ip = self.module.params['default_ip']
        self.tos_exp_dynamic = self.module.params['tos_exp_dynamic']
        self.tos_exp_static = self.module.params['tos_exp_static']
        self.damp_init_wait_time = self.module.params['damp_init_wait_time']
        self.damp_max_wait_time = self.module.params['damp_max_wait_time']
        self.damp_second_wait_time = self.module.params['damp_second_wait_time']
        self.delay_up_time = self.module.params['delay_up_time']
        self.state = self.module.params['state']
        self.host = self.module.params['host']
        self.username = self.module.params['username']
        self.port = self.module.params['port']
        self.changed = False
        self.bfd_dict = dict()
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def __init_module__(self):
        """init module"""
        required_together = [('damp_init_wait_time', 'damp_max_wait_time', 'damp_second_wait_time')]
        self.module = AnsibleModule(argument_spec=self.spec, required_together=required_together, supports_check_mode=True)

    def get_bfd_dict(self):
        """bfd config dict"""
        bfd_dict = dict()
        bfd_dict['global'] = dict()
        conf_str = CE_NC_GET_BFD % CE_NC_GET_BFD_GLB
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return bfd_dict
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        glb = root.find('bfd/bfdSchGlobal')
        if glb:
            for attr in glb:
                if attr.text is not None:
                    bfd_dict['global'][attr.tag] = attr.text
        return bfd_dict

    def config_global(self):
        """configures bfd global params"""
        xml_str = ''
        damp_chg = False
        if self.bfd_enable:
            if bool(self.bfd_dict['global'].get('bfdEnable', 'false') == 'true') != bool(self.bfd_enable == 'enable'):
                if self.bfd_enable == 'enable':
                    xml_str = '<bfdEnable>true</bfdEnable>'
                    self.updates_cmd.append('bfd')
                else:
                    xml_str = '<bfdEnable>false</bfdEnable>'
                    self.updates_cmd.append('undo bfd')
        bfd_state = 'disable'
        if self.bfd_enable:
            bfd_state = self.bfd_enable
        elif self.bfd_dict['global'].get('bfdEnable', 'false') == 'true':
            bfd_state = 'enable'
        if self.default_ip:
            if bfd_state == 'enable':
                if self.state == 'present' and self.default_ip != self.bfd_dict['global'].get('defaultIp'):
                    xml_str += '<defaultIp>%s</defaultIp>' % self.default_ip
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('default-ip-address %s' % self.default_ip)
                elif self.state == 'absent' and self.default_ip == self.bfd_dict['global'].get('defaultIp'):
                    xml_str += '<defaultIp/>'
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('undo default-ip-address')
        if self.tos_exp_dynamic is not None:
            if bfd_state == 'enable':
                if self.state == 'present' and self.tos_exp_dynamic != int(self.bfd_dict['global'].get('tosExp', '7')):
                    xml_str += '<tosExp>%s</tosExp>' % self.tos_exp_dynamic
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('tos-exp %s dynamic' % self.tos_exp_dynamic)
                elif self.state == 'absent' and self.tos_exp_dynamic == int(self.bfd_dict['global'].get('tosExp', '7')):
                    xml_str += '<tosExp/>'
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('undo tos-exp dynamic')
        if self.tos_exp_static is not None:
            if bfd_state == 'enable':
                if self.state == 'present' and self.tos_exp_static != int(self.bfd_dict['global'].get('tosExpStatic', '7')):
                    xml_str += '<tosExpStatic>%s</tosExpStatic>' % self.tos_exp_static
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('tos-exp %s static' % self.tos_exp_static)
                elif self.state == 'absent' and self.tos_exp_static == int(self.bfd_dict['global'].get('tosExpStatic', '7')):
                    xml_str += '<tosExpStatic/>'
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('undo tos-exp static')
        if self.delay_up_time is not None:
            if bfd_state == 'enable':
                delay_time = self.bfd_dict['global'].get('delayUpTimer', '0')
                if not delay_time or not delay_time.isdigit():
                    delay_time = '0'
                if self.state == 'present' and self.delay_up_time != int(delay_time):
                    xml_str += '<delayUpTimer>%s</delayUpTimer>' % self.delay_up_time
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('delay-up %s' % self.delay_up_time)
                elif self.state == 'absent' and self.delay_up_time == int(delay_time):
                    xml_str += '<delayUpTimer/>'
                    if 'bfd' not in self.updates_cmd:
                        self.updates_cmd.append('bfd')
                    self.updates_cmd.append('undo delay-up')
        if self.damp_init_wait_time is not None and self.damp_second_wait_time is not None and (self.damp_second_wait_time is not None):
            if bfd_state == 'enable':
                if self.state == 'present':
                    if self.damp_max_wait_time != int(self.bfd_dict['global'].get('dampMaxWaitTime', '2000')):
                        xml_str += '<dampMaxWaitTime>%s</dampMaxWaitTime>' % self.damp_max_wait_time
                        damp_chg = True
                    if self.damp_init_wait_time != int(self.bfd_dict['global'].get('dampInitWaitTime', '12000')):
                        xml_str += '<dampInitWaitTime>%s</dampInitWaitTime>' % self.damp_init_wait_time
                        damp_chg = True
                    if self.damp_second_wait_time != int(self.bfd_dict['global'].get('dampSecondWaitTime', '5000')):
                        xml_str += '<dampSecondWaitTime>%s</dampSecondWaitTime>' % self.damp_second_wait_time
                        damp_chg = True
                    if damp_chg:
                        if 'bfd' not in self.updates_cmd:
                            self.updates_cmd.append('bfd')
                        self.updates_cmd.append('dampening timer-interval maximum %s initial %s secondary %s' % (self.damp_max_wait_time, self.damp_init_wait_time, self.damp_second_wait_time))
                else:
                    damp_chg = True
                    if self.damp_max_wait_time != int(self.bfd_dict['global'].get('dampMaxWaitTime', '2000')):
                        damp_chg = False
                    if self.damp_init_wait_time != int(self.bfd_dict['global'].get('dampInitWaitTime', '12000')):
                        damp_chg = False
                    if self.damp_second_wait_time != int(self.bfd_dict['global'].get('dampSecondWaitTime', '5000')):
                        damp_chg = False
                    if damp_chg:
                        xml_str += '<dampMaxWaitTime/><dampInitWaitTime/><dampSecondWaitTime/>'
                        if 'bfd' not in self.updates_cmd:
                            self.updates_cmd.append('bfd')
                        self.updates_cmd.append('undo dampening timer-interval maximum %s initial %s secondary %s' % (self.damp_max_wait_time, self.damp_init_wait_time, self.damp_second_wait_time))
        if xml_str:
            return '<bfdSchGlobal operation="merge">' + xml_str + '</bfdSchGlobal>'
        else:
            return ''

    def netconf_load_config(self, xml_str):
        """load bfd config by netconf"""
        if not xml_str:
            return
        xml_cfg = '\n            <config>\n            <bfd xmlns="http://www.huawei.com/netconf/vrp" content-version="1.0" format-version="1.0">\n            %s\n            </bfd>\n            </config>' % xml_str
        set_nc_config(self.module, xml_cfg)
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if self.default_ip:
            if not check_default_ip(self.default_ip):
                self.module.fail_json(msg='Error: Default ip is invalid.')
        if self.tos_exp_dynamic is not None:
            if self.tos_exp_dynamic < 0 or self.tos_exp_dynamic > 7:
                self.module.fail_json(msg='Error: Session tos_exp_dynamic is not ranges from 0 to 7.')
        if self.tos_exp_static is not None:
            if self.tos_exp_static < 0 or self.tos_exp_static > 7:
                self.module.fail_json(msg='Error: Session tos_exp_static is not ranges from 0 to 7.')
        if self.damp_init_wait_time is not None:
            if self.damp_init_wait_time < 1 or self.damp_init_wait_time > 3600000:
                self.module.fail_json(msg='Error: Session damp_init_wait_time is not ranges from 1 to 3600000.')
        if self.damp_max_wait_time is not None:
            if self.damp_max_wait_time < 1 or self.damp_max_wait_time > 3600000:
                self.module.fail_json(msg='Error: Session damp_max_wait_time is not ranges from 1 to 3600000.')
        if self.damp_second_wait_time is not None:
            if self.damp_second_wait_time < 1 or self.damp_second_wait_time > 3600000:
                self.module.fail_json(msg='Error: Session damp_second_wait_time is not ranges from 1 to 3600000.')
        if self.delay_up_time is not None:
            if self.delay_up_time < 1 or self.delay_up_time > 600:
                self.module.fail_json(msg='Error: Session delay_up_time is not ranges from 1 to 600.')

    def get_proposed(self):
        """get proposed info"""
        self.proposed['bfd_enalbe'] = self.bfd_enable
        self.proposed['default_ip'] = self.default_ip
        self.proposed['tos_exp_dynamic'] = self.tos_exp_dynamic
        self.proposed['tos_exp_static'] = self.tos_exp_static
        self.proposed['damp_init_wait_time'] = self.damp_init_wait_time
        self.proposed['damp_max_wait_time'] = self.damp_max_wait_time
        self.proposed['damp_second_wait_time'] = self.damp_second_wait_time
        self.proposed['delay_up_time'] = self.delay_up_time
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if not self.bfd_dict:
            return
        self.existing['global'] = self.bfd_dict.get('global')

    def get_end_state(self):
        """get end state info"""
        bfd_dict = self.get_bfd_dict()
        if not bfd_dict:
            return
        self.end_state['global'] = bfd_dict.get('global')
        if self.existing == self.end_state:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        self.bfd_dict = self.get_bfd_dict()
        self.get_existing()
        self.get_proposed()
        xml_str = self.config_global()
        if xml_str:
            self.netconf_load_config(xml_str)
            self.changed = True
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