from __future__ import (absolute_import, division, print_function)
import copy
import re
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, set_nc_config, get_nc_config, execute_nc_action
class DldpInterface(object):
    """Manage interface dldp configuration"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.interface = self.module.params['interface']
        self.enable = self.module.params['enable'] or None
        self.reset = self.module.params['reset'] or None
        self.mode_enable = self.module.params['mode_enable'] or None
        self.local_mac = self.module.params['local_mac'] or None
        self.state = self.module.params['state']
        self.dldp_intf_conf = dict()
        self.same_conf = False
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = list()
        self.end_state = list()

    def check_config_if_same(self):
        """Judge whether current config is the same as what we excepted"""
        if self.state == 'absent':
            return False
        else:
            if self.enable and self.enable != self.dldp_intf_conf['dldpEnable']:
                return False
            if self.mode_enable and self.mode_enable != self.dldp_intf_conf['dldpCompatibleEnable']:
                return False
            if self.local_mac:
                flag = judge_is_mac_same(self.local_mac, self.dldp_intf_conf['dldpLocalMac'])
                if not flag:
                    return False
            if self.reset and self.reset == 'enable':
                return False
        return True

    def check_macaddr(self):
        """Check mac-address whether valid"""
        valid_char = '0123456789abcdef-'
        mac = self.local_mac
        if len(mac) > 16:
            return False
        mac_list = re.findall('([0-9a-fA-F]+)', mac)
        if len(mac_list) != 3:
            return False
        if mac.count('-') != 2:
            return False
        for dummy, value in enumerate(mac, start=0):
            if value.lower() not in valid_char:
                return False
        return True

    def check_params(self):
        """Check all input params"""
        if not self.interface:
            self.module.fail_json(msg='Error: Interface name cannot be empty.')
        if self.interface:
            intf_type = get_interface_type(self.interface)
            if not intf_type:
                self.module.fail_json(msg='Error: Interface name of %s is error.' % self.interface)
        if self.state == 'absent' and (self.reset or self.mode_enable or self.enable):
            self.module.fail_json(msg="Error: It's better to use state=present when configuring or unconfiguring enable, mode_enable or using reset flag. state=absent is just for when using local_mac param.")
        if self.state == 'absent' and (not self.local_mac):
            self.module.fail_json(msg='Error: Please specify local_mac parameter.')
        if self.state == 'present':
            if self.dldp_intf_conf['dldpEnable'] == 'disable' and (not self.enable) and (self.mode_enable or self.local_mac or self.reset):
                self.module.fail_json(msg='Error: when DLDP is already disabled on this port, mode_enable, local_mac and reset parameters are not expected to configure.')
            if self.enable == 'disable' and (self.mode_enable or self.local_mac or self.reset):
                self.module.fail_json(msg='Error: when using enable=disable, mode_enable, local_mac and reset parameters are not expected to configure.')
        if self.local_mac and (self.mode_enable == 'disable' or (self.dldp_intf_conf['dldpCompatibleEnable'] == 'disable' and self.mode_enable != 'enable')):
            self.module.fail_json(msg='Error: when DLDP compatible-mode is disabled on this port, Configuring local_mac is not allowed.')
        if self.local_mac:
            if not self.check_macaddr():
                self.module.fail_json(msg='Error: local_mac has invalid value %s.' % self.local_mac)

    def init_module(self):
        """Init module object"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed"""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_dldp_intf_exist_config(self):
        """Get current dldp existed config"""
        dldp_conf = dict()
        xml_str = CE_NC_GET_INTF_DLDP_CONFIG % self.interface
        con_obj = get_nc_config(self.module, xml_str)
        if '<data/>' in con_obj:
            dldp_conf['dldpEnable'] = 'disable'
            dldp_conf['dldpCompatibleEnable'] = ''
            dldp_conf['dldpLocalMac'] = ''
            return dldp_conf
        xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        topo = root.find('dldp/dldpInterfaces/dldpInterface')
        if topo is None:
            self.module.fail_json(msg='Error: Get current DLDP configuration failed.')
        for eles in topo:
            if eles.tag in ['dldpEnable', 'dldpCompatibleEnable', 'dldpLocalMac']:
                if not eles.text:
                    dldp_conf[eles.tag] = ''
                else:
                    if eles.tag == 'dldpEnable' or eles.tag == 'dldpCompatibleEnable':
                        if eles.text == 'true':
                            value = 'enable'
                        else:
                            value = 'disable'
                    else:
                        value = eles.text
                    dldp_conf[eles.tag] = value
        return dldp_conf

    def config_intf_dldp(self):
        """Config global dldp"""
        if self.same_conf:
            return
        if self.state == 'present':
            enable = self.enable
            if not self.enable:
                enable = self.dldp_intf_conf['dldpEnable']
            if enable == 'enable':
                enable = 'true'
            else:
                enable = 'false'
            mode_enable = self.mode_enable
            if not self.mode_enable:
                mode_enable = self.dldp_intf_conf['dldpCompatibleEnable']
            if mode_enable == 'enable':
                mode_enable = 'true'
            else:
                mode_enable = 'false'
            local_mac = self.local_mac
            if not self.local_mac:
                local_mac = self.dldp_intf_conf['dldpLocalMac']
            if self.enable == 'disable' and self.enable != self.dldp_intf_conf['dldpEnable']:
                xml_str = CE_NC_DELETE_DLDP_INTF_CONFIG % self.interface
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'DELETE_DLDP_INTF_CONFIG')
            elif self.dldp_intf_conf['dldpEnable'] == 'disable' and self.enable == 'enable':
                xml_str = CE_NC_CREATE_DLDP_INTF_CONFIG % (self.interface, 'true', mode_enable, local_mac)
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'CREATE_DLDP_INTF_CONFIG')
            elif self.dldp_intf_conf['dldpEnable'] == 'enable':
                if mode_enable == 'false':
                    local_mac = ''
                xml_str = CE_NC_MERGE_DLDP_INTF_CONFIG % (self.interface, enable, mode_enable, local_mac)
                ret_xml = set_nc_config(self.module, xml_str)
                self.check_response(ret_xml, 'MERGE_DLDP_INTF_CONFIG')
            if self.reset == 'enable':
                xml_str = CE_NC_ACTION_RESET_INTF_DLDP % self.interface
                ret_xml = execute_nc_action(self.module, xml_str)
                self.check_response(ret_xml, 'ACTION_RESET_INTF_DLDP')
            self.changed = True
        elif self.local_mac and judge_is_mac_same(self.local_mac, self.dldp_intf_conf['dldpLocalMac']):
            if self.dldp_intf_conf['dldpEnable'] == 'enable':
                dldp_enable = 'true'
            else:
                dldp_enable = 'false'
            if self.dldp_intf_conf['dldpCompatibleEnable'] == 'enable':
                dldp_compat_enable = 'true'
            else:
                dldp_compat_enable = 'false'
            xml_str = CE_NC_MERGE_DLDP_INTF_CONFIG % (self.interface, dldp_enable, dldp_compat_enable, '')
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'UNDO_DLDP_INTF_LOCAL_MAC_CONFIG')
            self.changed = True

    def get_existing(self):
        """Get existing info"""
        dldp_conf = dict()
        dldp_conf['interface'] = self.interface
        dldp_conf['enable'] = self.dldp_intf_conf.get('dldpEnable', None)
        dldp_conf['mode_enable'] = self.dldp_intf_conf.get('dldpCompatibleEnable', None)
        dldp_conf['local_mac'] = self.dldp_intf_conf.get('dldpLocalMac', None)
        dldp_conf['reset'] = 'disable'
        self.existing = copy.deepcopy(dldp_conf)

    def get_proposed(self):
        """Get proposed result """
        self.proposed = dict(interface=self.interface, enable=self.enable, mode_enable=self.mode_enable, local_mac=self.local_mac, reset=self.reset, state=self.state)

    def get_update_cmd(self):
        """Get updated commands"""
        if self.same_conf:
            return
        if self.state == 'present':
            if self.enable and self.enable != self.dldp_intf_conf['dldpEnable']:
                if self.enable == 'enable':
                    self.updates_cmd.append('dldp enable')
                elif self.enable == 'disable':
                    self.updates_cmd.append('undo dldp enable')
            if self.mode_enable and self.mode_enable != self.dldp_intf_conf['dldpCompatibleEnable']:
                if self.mode_enable == 'enable':
                    self.updates_cmd.append('dldp compatible-mode enable')
                else:
                    self.updates_cmd.append('undo dldp compatible-mode enable')
            if self.local_mac:
                flag = judge_is_mac_same(self.local_mac, self.dldp_intf_conf['dldpLocalMac'])
                if not flag:
                    self.updates_cmd.append('dldp compatible-mode local-mac %s' % self.local_mac)
            if self.reset and self.reset == 'enable':
                self.updates_cmd.append('dldp reset')
        elif self.changed:
            self.updates_cmd.append('undo dldp compatible-mode local-mac')

    def get_end_state(self):
        """Get end state info"""
        dldp_conf = dict()
        self.dldp_intf_conf = self.get_dldp_intf_exist_config()
        dldp_conf['interface'] = self.interface
        dldp_conf['enable'] = self.dldp_intf_conf.get('dldpEnable', None)
        dldp_conf['mode_enable'] = self.dldp_intf_conf.get('dldpCompatibleEnable', None)
        dldp_conf['local_mac'] = self.dldp_intf_conf.get('dldpLocalMac', None)
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
        """Execute task"""
        self.dldp_intf_conf = self.get_dldp_intf_exist_config()
        self.check_params()
        self.same_conf = self.check_config_if_same()
        self.get_existing()
        self.get_proposed()
        self.config_intf_dldp()
        self.get_update_cmd()
        self.get_end_state()
        self.show_result()