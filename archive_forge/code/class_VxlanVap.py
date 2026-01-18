from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class VxlanVap(object):
    """
    Manages VXLAN virtual access point.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.bridge_domain_id = self.module.params['bridge_domain_id']
        self.bind_vlan_id = self.module.params['bind_vlan_id']
        self.l2_sub_interface = self.module.params['l2_sub_interface']
        self.ce_vid = self.module.params['ce_vid']
        self.pe_vid = self.module.params['pe_vid']
        self.encapsulation = self.module.params['encapsulation']
        self.state = self.module.params['state']
        self.vap_info = dict()
        self.l2sub_info = dict()
        self.changed = False
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def __init_module__(self):
        """init module"""
        required_together = [()]
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_bd_vap_dict(self):
        """get virtual access point info"""
        vap_info = dict()
        conf_str = CE_NC_GET_BD_VAP % self.bridge_domain_id
        xml_str = get_nc_config(self.module, conf_str)
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        vap_info['bdId'] = self.bridge_domain_id
        root = ElementTree.fromstring(xml_str)
        vap_info['vlanList'] = ''
        vap_vlan = root.find('evc/bds/bd/bdBindVlan')
        if vap_vlan:
            for ele in vap_vlan:
                if ele.tag == 'vlanList':
                    vap_info['vlanList'] = ele.text
        vap_ifs = root.findall('evc/bds/bd/servicePoints/servicePoint/ifName')
        if_list = list()
        if vap_ifs:
            for vap_if in vap_ifs:
                if vap_if.tag == 'ifName':
                    if_list.append(vap_if.text)
        vap_info['intfList'] = if_list
        return vap_info

    def get_l2_sub_intf_dict(self, ifname):
        """get l2 sub-interface info"""
        intf_info = dict()
        if not ifname:
            return intf_info
        conf_str = CE_NC_GET_ENCAP % ifname
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return intf_info
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        bds = root.find('ethernet/servicePoints/servicePoint')
        if not bds:
            return intf_info
        for ele in bds:
            if ele.tag in ['ifName', 'flowType']:
                intf_info[ele.tag] = ele.text.lower()
        if intf_info.get('flowType') == 'dot1q':
            ce_vid = root.find('ethernet/servicePoints/servicePoint/flowDot1qs')
            intf_info['dot1qVids'] = ''
            if ce_vid:
                for ele in ce_vid:
                    if ele.tag == 'dot1qVids':
                        intf_info['dot1qVids'] = ele.text
        elif intf_info.get('flowType') == 'qinq':
            vids = root.find('ethernet/servicePoints/servicePoint/flowQinqs/flowQinq')
            if vids:
                for ele in vids:
                    if ele.tag in ['peVlanId', 'ceVids']:
                        intf_info[ele.tag] = ele.text
        return intf_info

    def config_traffic_encap_dot1q(self):
        """configure traffic encapsulation type dot1q"""
        xml_str = ''
        self.updates_cmd.append('interface %s' % self.l2_sub_interface)
        if self.state == 'present':
            if self.encapsulation != self.l2sub_info.get('flowType'):
                if self.ce_vid:
                    vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                    xml_str = CE_NC_SET_ENCAP_DOT1Q % (self.l2_sub_interface, vlan_bitmap, vlan_bitmap)
                    self.updates_cmd.append('encapsulation %s vid %s' % (self.encapsulation, self.ce_vid))
                else:
                    xml_str = CE_NC_SET_ENCAP % (self.l2_sub_interface, self.encapsulation)
                    self.updates_cmd.append('encapsulation %s' % self.encapsulation)
            elif self.ce_vid and (not is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('dot1qVids'))):
                vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                xml_str = CE_NC_SET_ENCAP_DOT1Q % (self.l2_sub_interface, vlan_bitmap, vlan_bitmap)
                self.updates_cmd.append('encapsulation %s vid %s' % (self.encapsulation, self.ce_vid))
        elif self.encapsulation == self.l2sub_info.get('flowType'):
            if self.ce_vid:
                if is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('dot1qVids')):
                    xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                    self.updates_cmd.append('undo encapsulation %s vid %s' % (self.encapsulation, self.ce_vid))
            else:
                xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                self.updates_cmd.append('undo encapsulation %s' % self.encapsulation)
        if not xml_str:
            self.updates_cmd.pop()
            return
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'CONFIG_INTF_ENCAP_DOT1Q')
        self.changed = True

    def config_traffic_encap_qinq(self):
        """configure traffic encapsulation type qinq"""
        xml_str = ''
        self.updates_cmd.append('interface %s' % self.l2_sub_interface)
        if self.state == 'present':
            if self.encapsulation != self.l2sub_info.get('flowType'):
                if self.ce_vid:
                    vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                    xml_str = CE_NC_SET_ENCAP_QINQ % (self.l2_sub_interface, self.pe_vid, vlan_bitmap, vlan_bitmap)
                    self.updates_cmd.append('encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
                else:
                    xml_str = CE_NC_SET_ENCAP % (self.l2_sub_interface, self.encapsulation)
                    self.updates_cmd.append('encapsulation %s' % self.encapsulation)
            elif self.ce_vid:
                if not is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('ceVids')) or self.pe_vid != self.l2sub_info.get('peVlanId'):
                    vlan_bitmap = vlan_vid_to_bitmap(self.ce_vid)
                    xml_str = CE_NC_SET_ENCAP_QINQ % (self.l2_sub_interface, self.pe_vid, vlan_bitmap, vlan_bitmap)
                    self.updates_cmd.append('encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
        elif self.encapsulation == self.l2sub_info.get('flowType'):
            if self.ce_vid:
                if is_vlan_in_bitmap(self.ce_vid, self.l2sub_info.get('ceVids')) and self.pe_vid == self.l2sub_info.get('peVlanId'):
                    xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                    self.updates_cmd.append('undo encapsulation %s vid %s ce-vid %s' % (self.encapsulation, self.pe_vid, self.ce_vid))
            else:
                xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                self.updates_cmd.append('undo encapsulation %s' % self.encapsulation)
        if not xml_str:
            self.updates_cmd.pop()
            return
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'CONFIG_INTF_ENCAP_QINQ')
        self.changed = True

    def config_traffic_encap(self):
        """configure traffic encapsulation types"""
        if not self.l2sub_info:
            self.module.fail_json(msg='Error: Interface %s does not exist.' % self.l2_sub_interface)
        if not self.encapsulation:
            return
        xml_str = ''
        if self.encapsulation in ['default', 'untag']:
            if self.state == 'present':
                if self.encapsulation != self.l2sub_info.get('flowType'):
                    xml_str = CE_NC_SET_ENCAP % (self.l2_sub_interface, self.encapsulation)
                    self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                    self.updates_cmd.append('encapsulation %s' % self.encapsulation)
            elif self.encapsulation == self.l2sub_info.get('flowType'):
                xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                self.updates_cmd.append('undo encapsulation %s' % self.encapsulation)
        elif self.encapsulation == 'none':
            if self.state == 'present':
                if self.encapsulation != self.l2sub_info.get('flowType'):
                    xml_str = CE_NC_UNSET_ENCAP % self.l2_sub_interface
                    self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                    self.updates_cmd.append('undo encapsulation %s' % self.l2sub_info.get('flowType'))
        elif self.encapsulation == 'dot1q':
            self.config_traffic_encap_dot1q()
            return
        elif self.encapsulation == 'qinq':
            self.config_traffic_encap_qinq()
            return
        else:
            pass
        if not xml_str:
            return
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'CONFIG_INTF_ENCAP')
        self.changed = True

    def config_vap_sub_intf(self):
        """configure a Layer 2 sub-interface as a service access point"""
        if not self.vap_info:
            self.module.fail_json(msg='Error: Bridge domain %s does not exist.' % self.bridge_domain_id)
        xml_str = ''
        if self.state == 'present':
            if self.l2_sub_interface not in self.vap_info['intfList']:
                self.updates_cmd.append('interface %s' % self.l2_sub_interface)
                self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
                xml_str = CE_NC_MERGE_BD_INTF % (self.bridge_domain_id, self.l2_sub_interface)
        elif self.l2_sub_interface in self.vap_info['intfList']:
            self.updates_cmd.append('interface %s' % self.l2_sub_interface)
            self.updates_cmd.append('undo bridge-domain %s' % self.bridge_domain_id)
            xml_str = CE_NC_DELETE_BD_INTF % (self.bridge_domain_id, self.l2_sub_interface)
        if not xml_str:
            return
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'CONFIG_VAP_SUB_INTERFACE')
        self.changed = True

    def config_vap_vlan(self):
        """configure a VLAN as a service access point"""
        xml_str = ''
        if self.state == 'present':
            if not is_vlan_in_bitmap(self.bind_vlan_id, self.vap_info['vlanList']):
                self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
                self.updates_cmd.append('l2 binding vlan %s' % self.bind_vlan_id)
                vlan_bitmap = vlan_vid_to_bitmap(self.bind_vlan_id)
                xml_str = CE_NC_MERGE_BD_VLAN % (self.bridge_domain_id, vlan_bitmap, vlan_bitmap)
        elif is_vlan_in_bitmap(self.bind_vlan_id, self.vap_info['vlanList']):
            self.updates_cmd.append('bridge-domain %s' % self.bridge_domain_id)
            self.updates_cmd.append('undo l2 binding vlan %s' % self.bind_vlan_id)
            vlan_bitmap = vlan_vid_to_bitmap(self.bind_vlan_id)
            xml_str = CE_NC_MERGE_BD_VLAN % (self.bridge_domain_id, '0' * 1024, vlan_bitmap)
        if not xml_str:
            return
        recv_xml = set_nc_config(self.module, xml_str)
        self.check_response(recv_xml, 'CONFIG_VAP_VLAN')
        self.changed = True

    def is_vlan_valid(self, vid, name):
        """check VLAN id"""
        if not vid:
            return
        if not vid.isdigit():
            self.module.fail_json(msg='Error: %s is not digit.' % name)
            return
        if int(vid) < 1 or int(vid) > 4094:
            self.module.fail_json(msg='Error: %s is not in the range from 1 to 4094.' % name)

    def is_l2_sub_intf_valid(self, ifname):
        """check l2 sub interface valid"""
        if ifname.count('.') != 1:
            return False
        if_num = ifname.split('.')[1]
        if not if_num.isdigit():
            return False
        if int(if_num) < 1 or int(if_num) > 4096:
            self.module.fail_json(msg='Error: Sub-interface number is not in the range from 1 to 4096.')
            return False
        if not get_interface_type(ifname):
            return False
        return True

    def check_params(self):
        """Check all input params"""
        if self.bridge_domain_id:
            if not self.bridge_domain_id.isdigit():
                self.module.fail_json(msg='Error: Bridge domain id is not digit.')
            if int(self.bridge_domain_id) < 1 or int(self.bridge_domain_id) > 16777215:
                self.module.fail_json(msg='Error: Bridge domain id is not in the range from 1 to 16777215.')
        if self.bind_vlan_id:
            self.is_vlan_valid(self.bind_vlan_id, 'bind_vlan_id')
        if self.l2_sub_interface and (not self.is_l2_sub_intf_valid(self.l2_sub_interface)):
            self.module.fail_json(msg='Error: l2_sub_interface is invalid.')
        if self.ce_vid:
            self.is_vlan_valid(self.ce_vid, 'ce_vid')
            if not self.encapsulation or self.encapsulation not in ['dot1q', 'qinq']:
                self.module.fail_json(msg="Error: ce_vid can not be set when encapsulation is '%s'." % self.encapsulation)
            if self.encapsulation == 'qinq' and (not self.pe_vid):
                self.module.fail_json(msg="Error: ce_vid and pe_vid must be set at the same time when encapsulation is '%s'." % self.encapsulation)
        if self.pe_vid:
            self.is_vlan_valid(self.pe_vid, 'pe_vid')
            if not self.encapsulation or self.encapsulation != 'qinq':
                self.module.fail_json(msg="Error: pe_vid can not be set when encapsulation is '%s'." % self.encapsulation)
            if not self.ce_vid:
                self.module.fail_json(msg="Error: ce_vid and pe_vid must be set at the same time when encapsulation is '%s'." % self.encapsulation)

    def get_proposed(self):
        """get proposed info"""
        if self.bridge_domain_id:
            self.proposed['bridge_domain_id'] = self.bridge_domain_id
        if self.bind_vlan_id:
            self.proposed['bind_vlan_id'] = self.bind_vlan_id
        if self.l2_sub_interface:
            self.proposed['l2_sub_interface'] = self.l2_sub_interface
        if self.encapsulation:
            self.proposed['encapsulation'] = self.encapsulation
        if self.ce_vid:
            self.proposed['ce_vid'] = self.ce_vid
        if self.pe_vid:
            self.proposed['pe_vid'] = self.pe_vid
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if self.bridge_domain_id:
            if self.bind_vlan_id or self.l2_sub_interface:
                self.existing['bridge_domain_id'] = self.bridge_domain_id
                self.existing['bind_vlan_list'] = bitmap_to_vlan_list(self.vap_info.get('vlanList'))
                self.existing['bind_intf_list'] = self.vap_info.get('intfList')
        if self.encapsulation and self.l2_sub_interface:
            self.existing['l2_sub_interface'] = self.l2_sub_interface
            self.existing['encapsulation'] = self.l2sub_info.get('flowType')
            if self.existing['encapsulation'] == 'dot1q':
                self.existing['ce_vid'] = bitmap_to_vlan_list(self.l2sub_info.get('dot1qVids'))
            if self.existing['encapsulation'] == 'qinq':
                self.existing['ce_vid'] = bitmap_to_vlan_list(self.l2sub_info.get('ceVids'))
                self.existing['pe_vid'] = self.l2sub_info.get('peVlanId')

    def get_end_state(self):
        """get end state info"""
        if self.bridge_domain_id:
            if self.bind_vlan_id or self.l2_sub_interface:
                vap_info = self.get_bd_vap_dict()
                self.end_state['bridge_domain_id'] = self.bridge_domain_id
                self.end_state['bind_vlan_list'] = bitmap_to_vlan_list(vap_info.get('vlanList'))
                self.end_state['bind_intf_list'] = vap_info.get('intfList')
        if self.encapsulation and self.l2_sub_interface:
            l2sub_info = self.get_l2_sub_intf_dict(self.l2_sub_interface)
            self.end_state['l2_sub_interface'] = self.l2_sub_interface
            self.end_state['encapsulation'] = l2sub_info.get('flowType')
            if self.end_state['encapsulation'] == 'dot1q':
                self.end_state['ce_vid'] = bitmap_to_vlan_list(l2sub_info.get('dot1qVids'))
            if self.end_state['encapsulation'] == 'qinq':
                self.end_state['ce_vid'] = bitmap_to_vlan_list(l2sub_info.get('ceVids'))
                self.end_state['pe_vid'] = l2sub_info.get('peVlanId')

    def data_init(self):
        """data init"""
        if self.l2_sub_interface:
            self.l2_sub_interface = self.l2_sub_interface.replace(' ', '').upper()
        if self.encapsulation and self.l2_sub_interface:
            self.l2sub_info = self.get_l2_sub_intf_dict(self.l2_sub_interface)
        if self.bridge_domain_id:
            if self.bind_vlan_id or self.l2_sub_interface:
                self.vap_info = self.get_bd_vap_dict()

    def work(self):
        """worker"""
        self.check_params()
        self.data_init()
        self.get_existing()
        self.get_proposed()
        if self.encapsulation and self.l2_sub_interface:
            self.config_traffic_encap()
        if self.bridge_domain_id:
            if self.l2_sub_interface:
                self.config_vap_sub_intf()
            if self.bind_vlan_id:
                self.config_vap_vlan()
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