from __future__ import (absolute_import, division, print_function)
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
class Dldp(object):
    """Manage global dldp configuration"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.enable = self.module.params['enable'] or None
        self.work_mode = self.module.params['work_mode'] or None
        self.internal = self.module.params['time_interval'] or None
        self.reset = self.module.params['reset'] or None
        self.auth_mode = self.module.params['auth_mode']
        self.auth_pwd = self.module.params['auth_pwd']
        self.dldp_conf = dict()
        self.same_conf = False
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = list()
        self.end_state = list()

    def check_config_if_same(self):
        """Judge whether current config is the same as what we excepted"""
        if self.enable and self.enable != self.dldp_conf['dldpEnable']:
            return False
        if self.internal and self.internal != self.dldp_conf['dldpInterval']:
            return False
        work_mode = 'normal'
        if self.dldp_conf['dldpWorkMode'] == 'dldpEnhance':
            work_mode = 'enhance'
        if self.work_mode and self.work_mode != work_mode:
            return False
        if self.auth_mode:
            if self.auth_mode != 'none':
                return False
            if self.auth_mode == 'none' and self.dldp_conf['dldpAuthMode'] != 'dldpAuthNone':
                return False
        if self.reset and self.reset == 'enable':
            return False
        return True

    def check_params(self):
        """Check all input params"""
        if self.auth_mode and self.auth_mode != 'none' and (not self.auth_pwd) or (self.auth_pwd and (not self.auth_mode)):
            self.module.fail_json(msg='Error: auth_mode and auth_pwd must both exist or not exist.')
        if self.dldp_conf['dldpEnable'] == 'disable' and (not self.enable):
            if self.work_mode or self.reset or self.internal or self.auth_mode:
                self.module.fail_json(msg='Error: when DLDP is already disabled globally, work_mode, time_internal auth_mode and reset parameters are not expected to configure.')
        if self.enable == 'disable' and (self.work_mode or self.internal or self.reset or self.auth_mode):
            self.module.fail_json(msg='Error: when using enable=disable, work_mode, time_internal auth_mode and reset parameters are not expected to configure.')
        if self.internal:
            if not self.internal.isdigit():
                self.module.fail_json(msg='Error: time_interval must be digit.')
            if int(self.internal) < 1 or int(self.internal) > 100:
                self.module.fail_json(msg='Error: The value of time_internal should be between 1 and 100.')
        if self.auth_pwd:
            if '?' in self.auth_pwd:
                self.module.fail_json(msg='Error: The auth_pwd string excludes a question mark (?).')
            if len(self.auth_pwd) != 24 and len(self.auth_pwd) != 32 and (len(self.auth_pwd) != 48) and (len(self.auth_pwd) != 108) and (len(self.auth_pwd) != 128):
                if len(self.auth_pwd) < 1 or len(self.auth_pwd) > 16:
                    self.module.fail_json(msg='Error: The value is a string of 1 to 16 case-sensitive plaintexts or 24/32/48/108/128 case-sensitive encrypted characters.')

    def init_module(self):
        """Init module object"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed"""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_dldp_exist_config(self):
        """Get current dldp existed configuration"""
        dldp_conf = dict()
        xml_str = CE_NC_GET_GLOBAL_DLDP_CONFIG
        con_obj = get_nc_config(self.module, xml_str)
        if '<data/>' in con_obj:
            return dldp_conf
        xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        topo = root.find('dldp/dldpSys')
        if not topo:
            self.module.fail_json(msg='Error: Get current DLDP configuration failed.')
        for eles in topo:
            if eles.tag in ['dldpEnable', 'dldpInterval', 'dldpWorkMode', 'dldpAuthMode']:
                if eles.tag == 'dldpEnable':
                    if eles.text == 'true':
                        value = 'enable'
                    else:
                        value = 'disable'
                else:
                    value = eles.text
                dldp_conf[eles.tag] = value
        return dldp_conf

    def config_global_dldp(self):
        """Config global dldp"""
        if self.same_conf:
            return
        enable = self.enable
        if not self.enable:
            enable = self.dldp_conf['dldpEnable']
        if enable == 'enable':
            enable = 'true'
        else:
            enable = 'false'
        internal = self.internal
        if not self.internal:
            internal = self.dldp_conf['dldpInterval']
        work_mode = self.work_mode
        if not self.work_mode:
            work_mode = self.dldp_conf['dldpWorkMode']
        if work_mode == 'enhance' or work_mode == 'dldpEnhance':
            work_mode = 'dldpEnhance'
        else:
            work_mode = 'dldpNormal'
        auth_mode = self.auth_mode
        if not self.auth_mode:
            auth_mode = self.dldp_conf['dldpAuthMode']
        if auth_mode == 'md5':
            auth_mode = 'dldpAuthMD5'
        elif auth_mode == 'simple':
            auth_mode = 'dldpAuthSimple'
        elif auth_mode == 'sha':
            auth_mode = 'dldpAuthSHA'
        elif auth_mode == 'hmac-sha256':
            auth_mode = 'dldpAuthHMAC-SHA256'
        elif auth_mode == 'none':
            auth_mode = 'dldpAuthNone'
        xml_str = CE_NC_MERGE_DLDP_GLOBAL_CONFIG_HEAD % (enable, internal, work_mode)
        if self.auth_mode:
            if self.auth_mode == 'none':
                xml_str += '<dldpAuthMode>dldpAuthNone</dldpAuthMode>'
            else:
                xml_str += '<dldpAuthMode>%s</dldpAuthMode>' % auth_mode
                xml_str += '<dldpPasswords>%s</dldpPasswords>' % self.auth_pwd
        xml_str += CE_NC_MERGE_DLDP_GLOBAL_CONFIG_TAIL
        ret_xml = set_nc_config(self.module, xml_str)
        self.check_response(ret_xml, 'MERGE_DLDP_GLOBAL_CONFIG')
        if self.reset == 'enable':
            xml_str = CE_NC_ACTION_RESET_DLDP
            ret_xml = execute_nc_action(self.module, xml_str)
            self.check_response(ret_xml, 'ACTION_RESET_DLDP')
        self.changed = True

    def get_existing(self):
        """Get existing info"""
        dldp_conf = dict()
        dldp_conf['enable'] = self.dldp_conf.get('dldpEnable', None)
        dldp_conf['time_interval'] = self.dldp_conf.get('dldpInterval', None)
        work_mode = self.dldp_conf.get('dldpWorkMode', None)
        if work_mode == 'dldpEnhance':
            dldp_conf['work_mode'] = 'enhance'
        else:
            dldp_conf['work_mode'] = 'normal'
        auth_mode = self.dldp_conf.get('dldpAuthMode', None)
        if auth_mode == 'dldpAuthNone':
            dldp_conf['auth_mode'] = 'none'
        elif auth_mode == 'dldpAuthSimple':
            dldp_conf['auth_mode'] = 'simple'
        elif auth_mode == 'dldpAuthMD5':
            dldp_conf['auth_mode'] = 'md5'
        elif auth_mode == 'dldpAuthSHA':
            dldp_conf['auth_mode'] = 'sha'
        else:
            dldp_conf['auth_mode'] = 'hmac-sha256'
        dldp_conf['reset'] = 'disable'
        self.existing = copy.deepcopy(dldp_conf)

    def get_proposed(self):
        """Get proposed result"""
        self.proposed = dict(enable=self.enable, work_mode=self.work_mode, time_interval=self.internal, reset=self.reset, auth_mode=self.auth_mode, auth_pwd=self.auth_pwd)

    def get_update_cmd(self):
        """Get update commands"""
        if self.same_conf:
            return
        if self.enable and self.enable != self.dldp_conf['dldpEnable']:
            if self.enable == 'enable':
                self.updates_cmd.append('dldp enable')
            elif self.enable == 'disable':
                self.updates_cmd.append('undo dldp enable')
                return
        work_mode = 'normal'
        if self.dldp_conf['dldpWorkMode'] == 'dldpEnhance':
            work_mode = 'enhance'
        if self.work_mode and self.work_mode != work_mode:
            if self.work_mode == 'enhance':
                self.updates_cmd.append('dldp work-mode enhance')
            else:
                self.updates_cmd.append('dldp work-mode normal')
        if self.internal and self.internal != self.dldp_conf['dldpInterval']:
            self.updates_cmd.append('dldp interval %s' % self.internal)
        if self.auth_mode:
            if self.auth_mode == 'none':
                self.updates_cmd.append('undo dldp authentication-mode')
            else:
                self.updates_cmd.append('dldp authentication-mode %s %s' % (self.auth_mode, self.auth_pwd))
        if self.reset and self.reset == 'enable':
            self.updates_cmd.append('dldp reset')

    def get_end_state(self):
        """Get end state info"""
        dldp_conf = dict()
        self.dldp_conf = self.get_dldp_exist_config()
        dldp_conf['enable'] = self.dldp_conf.get('dldpEnable', None)
        dldp_conf['time_interval'] = self.dldp_conf.get('dldpInterval', None)
        work_mode = self.dldp_conf.get('dldpWorkMode', None)
        if work_mode == 'dldpEnhance':
            dldp_conf['work_mode'] = 'enhance'
        else:
            dldp_conf['work_mode'] = 'normal'
        auth_mode = self.dldp_conf.get('dldpAuthMode', None)
        if auth_mode == 'dldpAuthNone':
            dldp_conf['auth_mode'] = 'none'
        elif auth_mode == 'dldpAuthSimple':
            dldp_conf['auth_mode'] = 'simple'
        elif auth_mode == 'dldpAuthMD5':
            dldp_conf['auth_mode'] = 'md5'
        elif auth_mode == 'dldpAuthSHA':
            dldp_conf['auth_mode'] = 'sha'
        else:
            dldp_conf['auth_mode'] = 'hmac-sha256'
        dldp_conf['reset'] = 'disable'
        if self.reset == 'enable':
            dldp_conf['reset'] = 'enable'
        self.end_state = copy.deepcopy(dldp_conf)

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
        """Worker"""
        self.dldp_conf = self.get_dldp_exist_config()
        self.check_params()
        self.same_conf = self.check_config_if_same()
        self.get_existing()
        self.get_proposed()
        self.config_global_dldp()
        self.get_update_cmd()
        self.get_end_state()
        self.show_result()