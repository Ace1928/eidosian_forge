from __future__ import (absolute_import, division, print_function)
import re
from xml.etree import ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class SwitchPort(object):
    """
    Manages Layer 2 switchport interfaces.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.interface = self.module.params['interface']
        self.mode = self.module.params['mode']
        self.state = self.module.params['state']
        self.default_vlan = self.module.params['default_vlan']
        self.pvid_vlan = self.module.params['pvid_vlan']
        self.trunk_vlans = self.module.params['trunk_vlans']
        self.untagged_vlans = self.module.params['untagged_vlans']
        self.tagged_vlans = self.module.params['tagged_vlans']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.intf_info = dict()
        self.intf_type = None

    def init_module(self):
        """ init module """
        required_if = [('state', 'absent', ['mode']), ('state', 'present', ['mode'])]
        mutually_exclusive = [['default_vlan', 'trunk_vlans'], ['default_vlan', 'pvid_vlan'], ['default_vlan', 'untagged_vlans'], ['trunk_vlans', 'untagged_vlans'], ['trunk_vlans', 'tagged_vlans'], ['default_vlan', 'tagged_vlans']]
        self.module = AnsibleModule(argument_spec=self.spec, required_if=required_if, supports_check_mode=True, mutually_exclusive=mutually_exclusive)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_interface_dict(self, ifname):
        """ get one interface attributes dict."""
        intf_info = dict()
        conf_str = CE_NC_GET_PORT_ATTR % ifname
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return intf_info
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        tree = ET.fromstring(xml_str)
        l2Enable = tree.find('ethernet/ethernetIfs/ethernetIf/l2Enable')
        intf_info['l2Enable'] = l2Enable.text
        port_type = tree.find('ethernet/ethernetIfs/ethernetIf/l2Attribute')
        for pre in port_type:
            intf_info[pre.tag] = pre.text
        intf_info['ifName'] = ifname
        if intf_info['trunkVlans'] is None:
            intf_info['trunkVlans'] = ''
        if intf_info['untagVlans'] is None:
            intf_info['untagVlans'] = ''
        return intf_info

    def is_l2switchport(self):
        """Check layer2 switch port"""
        return bool(self.intf_info['l2Enable'] == 'enable')

    def merge_access_vlan(self, ifname, default_vlan):
        """Merge access interface vlan"""
        change = False
        conf_str = ''
        self.updates_cmd.append('interface %s' % ifname)
        if self.state == 'present':
            if self.intf_info['linkType'] == 'access':
                if default_vlan and self.intf_info['pvid'] != default_vlan:
                    self.updates_cmd.append('port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'access', default_vlan, '', '')
                    change = True
            else:
                self.updates_cmd.append('port link-type access')
                if default_vlan:
                    self.updates_cmd.append('port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'access', default_vlan, '', '')
                else:
                    conf_str = CE_NC_SET_PORT % (ifname, 'access', '1', '', '')
                change = True
        elif self.state == 'absent':
            if self.intf_info['linkType'] == 'access':
                if default_vlan and self.intf_info['pvid'] == default_vlan and (default_vlan != '1'):
                    self.updates_cmd.append('undo port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'access', '1', '', '')
                    change = True
        if not change:
            self.updates_cmd.pop()
            return
        conf_str = '<config>' + conf_str + '</config>'
        rcv_xml = set_nc_config(self.module, conf_str)
        self.check_response(rcv_xml, 'MERGE_ACCESS_PORT')
        self.changed = True

    def merge_trunk_vlan(self, ifname, pvid_vlan, trunk_vlans):
        """Merge trunk interface vlan"""
        change = False
        xmlstr = ''
        pvid = ''
        trunk = ''
        self.updates_cmd.append('interface %s' % ifname)
        if trunk_vlans:
            vlan_list = self.vlan_range_to_list(trunk_vlans)
            vlan_map = self.vlan_list_to_bitmap(vlan_list)
        if self.state == 'present':
            if self.intf_info['linkType'] == 'trunk':
                if pvid_vlan and self.intf_info['pvid'] != pvid_vlan:
                    self.updates_cmd.append('port trunk pvid vlan %s' % pvid_vlan)
                    pvid = pvid_vlan
                    change = True
                if trunk_vlans:
                    add_vlans = self.vlan_bitmap_add(self.intf_info['trunkVlans'], vlan_map)
                    if not is_vlan_bitmap_empty(add_vlans):
                        self.updates_cmd.append('port trunk allow-pass %s' % trunk_vlans.replace(',', ' ').replace('-', ' to '))
                        trunk = '%s:%s' % (add_vlans, add_vlans)
                        change = True
                if pvid or trunk:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'trunk', pvid, trunk, '')
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not trunk:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
            else:
                self.updates_cmd.append('port link-type trunk')
                change = True
                if pvid_vlan:
                    self.updates_cmd.append('port trunk pvid vlan %s' % pvid_vlan)
                    pvid = pvid_vlan
                if trunk_vlans:
                    self.updates_cmd.append('port trunk allow-pass %s' % trunk_vlans.replace(',', ' ').replace('-', ' to '))
                    trunk = '%s:%s' % (vlan_map, vlan_map)
                if pvid or trunk:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'trunk', pvid, trunk, '')
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not trunk:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                if not pvid_vlan and (not trunk_vlans):
                    xmlstr += CE_NC_SET_PORT_MODE % (ifname, 'trunk')
                    self.updates_cmd.append('undo port trunk allow-pass vlan 1')
        elif self.state == 'absent':
            if self.intf_info['linkType'] == 'trunk':
                if pvid_vlan and self.intf_info['pvid'] == pvid_vlan and (pvid_vlan != '1'):
                    self.updates_cmd.append('undo port trunk pvid vlan %s' % pvid_vlan)
                    pvid = '1'
                    change = True
                if trunk_vlans:
                    del_vlans = self.vlan_bitmap_del(self.intf_info['trunkVlans'], vlan_map)
                    if not is_vlan_bitmap_empty(del_vlans):
                        self.updates_cmd.append('undo port trunk allow-pass %s' % trunk_vlans.replace(',', ' ').replace('-', ' to '))
                        undo_map = vlan_bitmap_undo(del_vlans)
                        trunk = '%s:%s' % (undo_map, del_vlans)
                        change = True
                if pvid or trunk:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'trunk', pvid, trunk, '')
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not trunk:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
        if not change:
            self.updates_cmd.pop()
            return
        conf_str = '<config>' + xmlstr + '</config>'
        rcv_xml = set_nc_config(self.module, conf_str)
        self.check_response(rcv_xml, 'MERGE_TRUNK_PORT')
        self.changed = True

    def merge_hybrid_vlan(self, ifname, pvid_vlan, tagged_vlans, untagged_vlans):
        """Merge hybrid interface vlan"""
        change = False
        xmlstr = ''
        pvid = ''
        tagged = ''
        untagged = ''
        self.updates_cmd.append('interface %s' % ifname)
        if tagged_vlans:
            vlan_targed_list = self.vlan_range_to_list(tagged_vlans)
            vlan_targed_map = self.vlan_list_to_bitmap(vlan_targed_list)
        if untagged_vlans:
            vlan_untarged_list = self.vlan_range_to_list(untagged_vlans)
            vlan_untarged_map = self.vlan_list_to_bitmap(vlan_untarged_list)
        if self.state == 'present':
            if self.intf_info['linkType'] == 'hybrid':
                if pvid_vlan and self.intf_info['pvid'] != pvid_vlan:
                    self.updates_cmd.append('port hybrid pvid vlan %s' % pvid_vlan)
                    pvid = pvid_vlan
                    change = True
                if tagged_vlans:
                    add_vlans = self.vlan_bitmap_add(self.intf_info['trunkVlans'], vlan_targed_map)
                    if not is_vlan_bitmap_empty(add_vlans):
                        self.updates_cmd.append('port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                        tagged = '%s:%s' % (add_vlans, add_vlans)
                        change = True
                if untagged_vlans:
                    add_vlans = self.vlan_bitmap_add(self.intf_info['untagVlans'], vlan_untarged_map)
                    if not is_vlan_bitmap_empty(add_vlans):
                        self.updates_cmd.append('port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                        untagged = '%s:%s' % (add_vlans, add_vlans)
                        change = True
                if pvid or tagged or untagged:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not tagged:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                    if not untagged:
                        xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
            else:
                self.updates_cmd.append('port link-type hybrid')
                change = True
                if pvid_vlan:
                    self.updates_cmd.append('port hybrid pvid vlan %s' % pvid_vlan)
                    pvid = pvid_vlan
                if tagged_vlans:
                    self.updates_cmd.append('port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                    tagged = '%s:%s' % (vlan_targed_map, vlan_targed_map)
                if untagged_vlans:
                    self.updates_cmd.append('port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                    untagged = '%s:%s' % (vlan_untarged_map, vlan_untarged_map)
                if pvid or tagged or untagged:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not tagged:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                    if not untagged:
                        xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
                if not pvid_vlan and (not tagged_vlans) and (not untagged_vlans):
                    xmlstr += CE_NC_SET_PORT_MODE % (ifname, 'hybrid')
                    self.updates_cmd.append('undo port hybrid untagged vlan 1')
        elif self.state == 'absent':
            if self.intf_info['linkType'] == 'hybrid':
                if pvid_vlan and self.intf_info['pvid'] == pvid_vlan and (pvid_vlan != '1'):
                    self.updates_cmd.append('undo port hybrid pvid vlan %s' % pvid_vlan)
                    pvid = '1'
                    change = True
                if tagged_vlans:
                    del_vlans = self.vlan_bitmap_del(self.intf_info['trunkVlans'], vlan_targed_map)
                    if not is_vlan_bitmap_empty(del_vlans):
                        self.updates_cmd.append('undo port hybrid tagged vlan %s' % tagged_vlans.replace(',', ' ').replace('-', ' to '))
                        undo_map = vlan_bitmap_undo(del_vlans)
                        tagged = '%s:%s' % (undo_map, del_vlans)
                        change = True
                if untagged_vlans:
                    del_vlans = self.vlan_bitmap_del(self.intf_info['untagVlans'], vlan_untarged_map)
                    if not is_vlan_bitmap_empty(del_vlans):
                        self.updates_cmd.append('undo port hybrid untagged vlan %s' % untagged_vlans.replace(',', ' ').replace('-', ' to '))
                        undo_map = vlan_bitmap_undo(del_vlans)
                        untagged = '%s:%s' % (undo_map, del_vlans)
                        change = True
                if pvid or tagged or untagged:
                    xmlstr += CE_NC_SET_PORT % (ifname, 'hybrid', pvid, tagged, untagged)
                    if not pvid:
                        xmlstr = xmlstr.replace('<pvid></pvid>', '')
                    if not tagged:
                        xmlstr = xmlstr.replace('<trunkVlans></trunkVlans>', '')
                    if not untagged:
                        xmlstr = xmlstr.replace('<untagVlans></untagVlans>', '')
        if not change:
            self.updates_cmd.pop()
            return
        conf_str = '<config>' + xmlstr + '</config>'
        rcv_xml = set_nc_config(self.module, conf_str)
        self.check_response(rcv_xml, 'MERGE_HYBRID_PORT')
        self.changed = True

    def merge_dot1qtunnel_vlan(self, ifname, default_vlan):
        """Merge dot1qtunnel"""
        change = False
        conf_str = ''
        self.updates_cmd.append('interface %s' % ifname)
        if self.state == 'present':
            if self.intf_info['linkType'] == 'dot1qtunnel':
                if default_vlan and self.intf_info['pvid'] != default_vlan:
                    self.updates_cmd.append('port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', default_vlan, '', '')
                    change = True
            else:
                self.updates_cmd.append('port link-type dot1qtunnel')
                if default_vlan:
                    self.updates_cmd.append('port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', default_vlan, '', '')
                else:
                    conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', '1', '', '')
                change = True
        elif self.state == 'absent':
            if self.intf_info['linkType'] == 'dot1qtunnel':
                if default_vlan and self.intf_info['pvid'] == default_vlan and (default_vlan != '1'):
                    self.updates_cmd.append('undo port default vlan %s' % default_vlan)
                    conf_str = CE_NC_SET_PORT % (ifname, 'dot1qtunnel', '1', '', '')
                    change = True
        if not change:
            self.updates_cmd.pop()
            return
        conf_str = '<config>' + conf_str + '</config>'
        rcv_xml = set_nc_config(self.module, conf_str)
        self.check_response(rcv_xml, 'MERGE_DOT1QTUNNEL_PORT')
        self.changed = True

    def default_switchport(self, ifname):
        """Set interface default or unconfigured"""
        change = False
        if self.intf_info['linkType'] != 'access':
            self.updates_cmd.append('interface %s' % ifname)
            self.updates_cmd.append('port link-type access')
            self.updates_cmd.append('port default vlan 1')
            change = True
        elif self.intf_info['pvid'] != '1':
            self.updates_cmd.append('interface %s' % ifname)
            self.updates_cmd.append('port default vlan 1')
            change = True
        if not change:
            return
        conf_str = CE_NC_SET_DEFAULT_PORT % ifname
        rcv_xml = set_nc_config(self.module, conf_str)
        self.check_response(rcv_xml, 'DEFAULT_INTF_VLAN')
        self.changed = True

    def vlan_series(self, vlanid_s):
        """ convert vlan range to vlan list """
        vlan_list = []
        peerlistlen = len(vlanid_s)
        if peerlistlen != 2:
            self.module.fail_json(msg='Error: Format of vlanid is invalid.')
        for num in range(peerlistlen):
            if not vlanid_s[num].isdigit():
                self.module.fail_json(msg='Error: Format of vlanid is invalid.')
        if int(vlanid_s[0]) > int(vlanid_s[1]):
            self.module.fail_json(msg='Error: Format of vlanid is invalid.')
        elif int(vlanid_s[0]) == int(vlanid_s[1]):
            vlan_list.append(str(vlanid_s[0]))
            return vlan_list
        for num in range(int(vlanid_s[0]), int(vlanid_s[1])):
            vlan_list.append(str(num))
        vlan_list.append(vlanid_s[1])
        return vlan_list

    def vlan_region(self, vlanid_list):
        """ convert vlan range to vlan list """
        vlan_list = []
        peerlistlen = len(vlanid_list)
        for num in range(peerlistlen):
            if vlanid_list[num].isdigit():
                vlan_list.append(vlanid_list[num])
            else:
                vlan_s = self.vlan_series(vlanid_list[num].split('-'))
                vlan_list.extend(vlan_s)
        return vlan_list

    def vlan_range_to_list(self, vlan_range):
        """ convert vlan range to vlan list """
        vlan_list = self.vlan_region(vlan_range.split(','))
        return vlan_list

    def vlan_list_to_bitmap(self, vlanlist):
        """ convert vlan list to vlan bitmap """
        vlan_bit = ['0'] * 1024
        bit_int = [0] * 1024
        vlan_list_len = len(vlanlist)
        for num in range(vlan_list_len):
            tagged_vlans = int(vlanlist[num])
            if tagged_vlans <= 0 or tagged_vlans > 4094:
                self.module.fail_json(msg='Error: Vlan id is not in the range from 1 to 4094.')
            j = tagged_vlans // 4
            bit_int[j] |= 8 >> tagged_vlans % 4
            vlan_bit[j] = hex(bit_int[j])[2]
        vlan_xml = ''.join(vlan_bit)
        return vlan_xml

    def bitmap_to_vlan_list(self, bitmap):
        """convert VLAN bitmap to VLAN list"""
        vlan_list = list()
        if not bitmap:
            return vlan_list
        for i in range(len(bitmap)):
            if bitmap[i] == '0':
                continue
            bit = int(bitmap[i], 16)
            if bit & 8:
                vlan_list.append(str(i * 4))
            if bit & 4:
                vlan_list.append(str(i * 4 + 1))
            if bit & 2:
                vlan_list.append(str(i * 4 + 2))
            if bit & 1:
                vlan_list.append(str(i * 4 + 3))
        return vlan_list

    def vlan_bitmap_add(self, oldmap, newmap):
        """vlan add bitmap"""
        vlan_bit = ['0'] * 1024
        if len(newmap) != 1024:
            self.module.fail_json(msg='Error: New vlan bitmap is invalid.')
        if len(oldmap) != 1024 and len(oldmap) != 0:
            self.module.fail_json(msg='Error: old vlan bitmap is invalid.')
        if len(oldmap) == 0:
            return newmap
        for num in range(1024):
            new_tmp = int(newmap[num], 16)
            old_tmp = int(oldmap[num], 16)
            add = ~(new_tmp & old_tmp) & new_tmp
            vlan_bit[num] = hex(add)[2]
        vlan_xml = ''.join(vlan_bit)
        return vlan_xml

    def vlan_bitmap_del(self, oldmap, delmap):
        """vlan del bitmap"""
        vlan_bit = ['0'] * 1024
        if not oldmap or len(oldmap) == 0:
            return ''.join(vlan_bit)
        if len(oldmap) != 1024 or len(delmap) != 1024:
            self.module.fail_json(msg='Error: vlan bitmap is invalid.')
        for num in range(1024):
            tmp = int(delmap[num], 16) & int(oldmap[num], 16)
            vlan_bit[num] = hex(tmp)[2]
        vlan_xml = ''.join(vlan_bit)
        return vlan_xml

    def check_params(self):
        """Check all input params"""
        if self.interface:
            self.intf_type = get_interface_type(self.interface)
            if not self.intf_type:
                self.module.fail_json(msg='Error: Interface name of %s is error.' % self.interface)
        if not self.intf_type or not is_portswitch_enalbed(self.intf_type):
            self.module.fail_json(msg='Error: Interface %s is error.')
        if self.default_vlan:
            if not self.default_vlan.isdigit():
                self.module.fail_json(msg='Error: Access vlan id is invalid.')
            if int(self.default_vlan) <= 0 or int(self.default_vlan) > 4094:
                self.module.fail_json(msg='Error: Access vlan id is not in the range from 1 to 4094.')
        if self.pvid_vlan:
            if not self.pvid_vlan.isdigit():
                self.module.fail_json(msg='Error: Pvid vlan id is invalid.')
            if int(self.pvid_vlan) <= 0 or int(self.pvid_vlan) > 4094:
                self.module.fail_json(msg='Error: Pvid vlan id is not in the range from 1 to 4094.')
        self.intf_info = self.get_interface_dict(self.interface)
        if not self.intf_info:
            self.module.fail_json(msg='Error: Interface does not exist.')
        if not self.is_l2switchport():
            self.module.fail_json(msg='Error: Interface is not layer2 switch port.')
        if self.state == 'unconfigured':
            if any([self.mode, self.default_vlan, self.pvid_vlan, self.trunk_vlans, self.untagged_vlans, self.tagged_vlans]):
                self.module.fail_json(msg='Error: When state is unconfigured, only interface name exists.')
        elif self.mode == 'access':
            if any([self.pvid_vlan, self.trunk_vlans, self.untagged_vlans, self.tagged_vlans]):
                self.module.fail_json(msg='Error: When mode is access, only default_vlan can be supported.')
        elif self.mode == 'trunk':
            if any([self.default_vlan, self.untagged_vlans, self.tagged_vlans]):
                self.module.fail_json(msg='Error: When mode is trunk, only pvid_vlan and trunk_vlans can exist.')
        elif self.mode == 'hybrid':
            if any([self.default_vlan, self.trunk_vlans]):
                self.module.fail_json(msg='Error: When mode is hybrid, default_vlan and trunk_vlans cannot exist')
        elif any([self.pvid_vlan, self.trunk_vlans, self.untagged_vlans, self.tagged_vlans]):
            self.module.fail_json(msg='Error: When mode is dot1qtunnel, only default_vlan can be supported.')

    def get_proposed(self):
        """get proposed info"""
        self.proposed['state'] = self.state
        self.proposed['interface'] = self.interface
        self.proposed['mode'] = self.mode
        if self.mode:
            if self.mode == 'access':
                self.proposed['access_pvid'] = self.default_vlan
            elif self.mode == 'trunk':
                self.proposed['pvid_vlan'] = self.pvid_vlan
                self.proposed['trunk_vlans'] = self.trunk_vlans
            elif self.mode == 'hybrid':
                self.proposed['pvid_vlan'] = self.pvid_vlan
                self.proposed['untagged_vlans'] = self.untagged_vlans
                self.proposed['tagged_vlans'] = self.tagged_vlans
            else:
                self.proposed['dot1qtunnel_pvid'] = self.default_vlan

    def get_existing(self):
        """get existing info"""
        if self.intf_info:
            self.existing['interface'] = self.intf_info['ifName']
            self.existing['switchport'] = self.intf_info['l2Enable']
            self.existing['mode'] = self.intf_info['linkType']
            if self.intf_info['linkType'] == 'access':
                self.existing['access_pvid'] = self.intf_info['pvid']
            elif self.intf_info['linkType'] == 'trunk':
                self.existing['trunk_pvid'] = self.intf_info['pvid']
                self.existing['trunk_vlans'] = self.bitmap_to_vlan_list(self.intf_info['trunkVlans'])
            elif self.intf_info['linkType'] == 'hybrid':
                self.existing['hybrid_pvid'] = self.intf_info['pvid']
                self.existing['hybrid_untagged_vlans'] = self.intf_info['untagVlans']
                self.existing['hybrid_tagged_vlans'] = self.bitmap_to_vlan_list(self.intf_info['trunkVlans'])
            else:
                self.existing['dot1qtunnel_pvid'] = self.intf_info['pvid']

    def get_end_state(self):
        """get end state info"""
        end_info = self.get_interface_dict(self.interface)
        if end_info:
            self.end_state['interface'] = end_info['ifName']
            self.end_state['switchport'] = end_info['l2Enable']
            self.end_state['mode'] = end_info['linkType']
            if end_info['linkType'] == 'access':
                self.end_state['access_pvid'] = end_info['pvid']
            elif end_info['linkType'] == 'trunk':
                self.end_state['trunk_pvid'] = end_info['pvid']
                self.end_state['trunk_vlans'] = self.bitmap_to_vlan_list(end_info['trunkVlans'])
            elif end_info['linkType'] == 'hybrid':
                self.end_state['hybrid_pvid'] = end_info['pvid']
                self.end_state['hybrid_untagged_vlans'] = end_info['untagVlans']
                self.end_state['hybrid_tagged_vlans'] = self.bitmap_to_vlan_list(end_info['trunkVlans'])
            else:
                self.end_state['dot1qtunnel_pvid'] = end_info['pvid']
        if self.end_state == self.existing:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        if not self.intf_info:
            self.module.fail_json(msg='Error: interface does not exist.')
        self.get_existing()
        self.get_proposed()
        if self.state == 'present' or self.state == 'absent':
            if self.mode == 'access':
                self.merge_access_vlan(self.interface, self.default_vlan)
            elif self.mode == 'trunk':
                self.merge_trunk_vlan(self.interface, self.pvid_vlan, self.trunk_vlans)
            elif self.mode == 'hybrid':
                self.merge_hybrid_vlan(self.interface, self.pvid_vlan, self.tagged_vlans, self.untagged_vlans)
            else:
                self.merge_dot1qtunnel_vlan(self.interface, self.default_vlan)
        else:
            self.default_switchport(self.interface)
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