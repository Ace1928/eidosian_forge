from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config, set_nc_config
class VrfInterface(object):
    """Manage vpn instance"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.vrf = self.module.params['vrf']
        self.vpn_interface = self.module.params['vpn_interface']
        self.vpn_interface = self.vpn_interface.upper().replace(' ', '')
        self.state = self.module.params['state']
        self.intf_info = dict()
        self.intf_info['isL2SwitchPort'] = None
        self.intf_info['vrfName'] = None
        self.conf_exist = False
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def init_module(self):
        """init_module"""
        required_one_of = [('vrf', 'vpn_interface')]
        self.module = AnsibleModule(argument_spec=self.spec, required_one_of=required_one_of, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_update_cmd(self):
        """ get  updated command"""
        if self.conf_exist:
            return
        if self.state == 'absent':
            self.updates_cmd.append('undo ip binding vpn-instance %s' % self.vrf)
            return
        if self.vrf != self.intf_info['vrfName']:
            self.updates_cmd.append('ip binding vpn-instance %s' % self.vrf)
        return

    def check_params(self):
        """Check all input params"""
        if not self.is_vrf_exist():
            self.module.fail_json(msg='Error: The VPN instance is not existed.')
        if self.state == 'absent':
            if self.vrf != self.intf_info['vrfName']:
                self.module.fail_json(msg='Error: The VPN instance is not bound to the interface.')
        if self.intf_info['isL2SwitchPort'] == 'true':
            self.module.fail_json(msg='Error: L2Switch Port can not binding a VPN instance.')
        if self.vpn_interface:
            intf_type = get_interface_type(self.vpn_interface)
            if not intf_type:
                self.module.fail_json(msg='Error: interface name of %s is error.' % self.vpn_interface)
        if self.vrf == '_public_':
            self.module.fail_json(msg='Error: The vrf name _public_ is reserved.')
        if len(self.vrf) < 1 or len(self.vrf) > 31:
            self.module.fail_json(msg='Error: The vrf name length must be between 1 and 31.')

    def get_interface_vpn_name(self, vpninfo, vpn_name):
        """ get vpn instance name"""
        l3vpn_if = vpninfo.findall('l3vpnIf')
        for l3vpn_ifinfo in l3vpn_if:
            for ele in l3vpn_ifinfo:
                if ele.tag in ['ifName']:
                    if ele.text.lower() == self.vpn_interface.lower():
                        self.intf_info['vrfName'] = vpn_name

    def get_interface_vpn(self):
        """ get the VPN instance associated with the interface"""
        xml_str = CE_NC_GET_VRF_INTERFACE
        con_obj = get_nc_config(self.module, xml_str)
        if '<data/>' in con_obj:
            return
        xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        vpns = root.findall('l3vpn/l3vpncomm/l3vpnInstances/l3vpnInstance')
        if vpns:
            for vpnele in vpns:
                vpn_name = None
                for vpninfo in vpnele:
                    if vpninfo.tag == 'vrfName':
                        vpn_name = vpninfo.text
                    if vpninfo.tag == 'l3vpnIfs':
                        self.get_interface_vpn_name(vpninfo, vpn_name)
        return

    def is_vrf_exist(self):
        """ judge whether the VPN instance is existed"""
        conf_str = CE_NC_GET_VRF % self.vrf
        con_obj = get_nc_config(self.module, conf_str)
        if '<data/>' in con_obj:
            return False
        return True

    def get_intf_conf_info(self):
        """ get related configuration of the interface"""
        conf_str = CE_NC_GET_INTF % self.vpn_interface
        con_obj = get_nc_config(self.module, conf_str)
        if '<data/>' in con_obj:
            return
        xml_str = con_obj.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        interface = root.find('ifm/interfaces/interface')
        if interface:
            for eles in interface:
                if eles.tag in ['isL2SwitchPort']:
                    self.intf_info[eles.tag] = eles.text
        self.get_interface_vpn()
        return

    def get_existing(self):
        """get existing config"""
        self.existing = dict(vrf=self.intf_info['vrfName'], vpn_interface=self.vpn_interface)

    def get_proposed(self):
        """get_proposed"""
        self.proposed = dict(vrf=self.vrf, vpn_interface=self.vpn_interface, state=self.state)

    def get_end_state(self):
        """get_end_state"""
        self.intf_info['vrfName'] = None
        self.get_intf_conf_info()
        self.end_state = dict(vrf=self.intf_info['vrfName'], vpn_interface=self.vpn_interface)

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
        if self.state == 'absent':
            return False
        delta = set(self.proposed.items()).difference(self.existing.items())
        delta = dict(delta)
        if len(delta) == 1 and delta['state']:
            return True
        return False

    def config_interface_vrf(self):
        """ configure VPN instance of the interface"""
        if not self.conf_exist and self.state == 'present':
            xml_str = CE_NC_MERGE_VRF_INTERFACE % (self.vrf, self.vpn_interface)
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'VRF_INTERFACE_CONFIG')
            self.changed = True
        elif self.state == 'absent':
            xml_str = CE_NC_DEL_INTF_VPN % (self.vrf, self.vpn_interface)
            ret_xml = set_nc_config(self.module, xml_str)
            self.check_response(ret_xml, 'DEL_VRF_INTERFACE_CONFIG')
            self.changed = True

    def work(self):
        """execute task"""
        self.get_intf_conf_info()
        self.check_params()
        self.get_existing()
        self.get_proposed()
        self.conf_exist = self.judge_if_config_exist()
        self.config_interface_vrf()
        self.get_update_cmd()
        self.get_end_state()
        self.show_result()