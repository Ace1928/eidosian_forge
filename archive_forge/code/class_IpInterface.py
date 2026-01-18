from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class IpInterface(object):
    """
    Manages L3 attributes for IPv4 and IPv6 interfaces.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.interface = self.module.params['interface']
        self.addr = self.module.params['addr']
        self.mask = self.module.params['mask']
        self.version = self.module.params['version']
        self.ipv4_type = self.module.params['ipv4_type']
        self.state = self.module.params['state']
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.intf_info = dict()
        self.intf_type = None

    def __init_module__(self):
        """ init module """
        required_if = [('version', 'v4', ('addr', 'mask'))]
        required_together = [('addr', 'mask')]
        self.module = AnsibleModule(argument_spec=self.spec, required_if=required_if, required_together=required_together, supports_check_mode=True)

    def netconf_set_config(self, xml_str, xml_name):
        """ netconf set config """
        rcv_xml = set_nc_config(self.module, xml_str)
        if '<ok/>' not in rcv_xml:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_interface_dict(self, ifname):
        """ get one interface attributes dict."""
        intf_info = dict()
        conf_str = CE_NC_GET_INTF % ifname
        rcv_xml = get_nc_config(self.module, conf_str)
        if '<data/>' in rcv_xml:
            return intf_info
        intf = re.findall('.*<ifName>(.*)</ifName>.*\\s*<isL2SwitchPort>(.*)</isL2SwitchPort>.*', rcv_xml)
        if intf:
            intf_info = dict(ifName=intf[0][0], isL2SwitchPort=intf[0][1])
        ipv4_info = re.findall('.*<ifIpAddr>(.*)</ifIpAddr>.*\\s*<subnetMask>(.*)</subnetMask>.*\\s*<addrType>(.*)</addrType>.*', rcv_xml)
        intf_info['am4CfgAddr'] = list()
        for info in ipv4_info:
            intf_info['am4CfgAddr'].append(dict(ifIpAddr=info[0], subnetMask=info[1], addrType=info[2]))
        ipv6_info = re.findall('.*<ifmAm6>.*\\s*<enableFlag>(.*)</enableFlag>.*', rcv_xml)
        if not ipv6_info:
            self.module.fail_json(msg='Error: Fail to get interface %s IPv6 state.' % self.interface)
        else:
            intf_info['enableFlag'] = ipv6_info[0]
        ipv6_info = re.findall('.*<ifIp6Addr>(.*)</ifIp6Addr>.*\\s*<addrPrefixLen>(.*)</addrPrefixLen>.*\\s*<addrType6>(.*)</addrType6>.*', rcv_xml)
        intf_info['am6CfgAddr'] = list()
        for info in ipv6_info:
            intf_info['am6CfgAddr'].append(dict(ifIp6Addr=info[0], addrPrefixLen=info[1], addrType6=info[2]))
        return intf_info

    def convert_len_to_mask(self, masklen):
        """convert mask length to ip address mask, i.e. 24 to 255.255.255.0"""
        mask_int = ['0'] * 4
        length = int(masklen)
        if length > 32:
            self.module.fail_json(msg='Error: IPv4 ipaddress mask length is invalid.')
        if length < 8:
            mask_int[0] = str(int(255 << 8 - length % 8 & 255))
        if length >= 8:
            mask_int[0] = '255'
            mask_int[1] = str(int(255 << 16 - length % 16 & 255))
        if length >= 16:
            mask_int[1] = '255'
            mask_int[2] = str(int(255 << 24 - length % 24 & 255))
        if length >= 24:
            mask_int[2] = '255'
            mask_int[3] = str(int(255 << 32 - length % 32 & 255))
        if length == 32:
            mask_int[3] = '255'
        return '.'.join(mask_int)

    def is_ipv4_exist(self, addr, maskstr, ipv4_type):
        """"Check IPv4 address exist"""
        addrs = self.intf_info['am4CfgAddr']
        if not addrs:
            return False
        for address in addrs:
            if address['ifIpAddr'] == addr:
                return address['subnetMask'] == maskstr and address['addrType'] == ipv4_type
        return False

    def get_ipv4_main_addr(self):
        """get IPv4 main address"""
        addrs = self.intf_info['am4CfgAddr']
        if not addrs:
            return None
        for address in addrs:
            if address['addrType'] == 'main':
                return address
        return None

    def is_ipv6_exist(self, addr, masklen):
        """Check IPv6 address exist"""
        addrs = self.intf_info['am6CfgAddr']
        if not addrs:
            return False
        for address in addrs:
            if address['ifIp6Addr'] == addr.upper():
                if address['addrPrefixLen'] == masklen and address['addrType6'] == 'global':
                    return True
                else:
                    self.module.fail_json(msg='Error: Input IPv6 address or mask is invalid.')
        return False

    def set_ipv4_addr(self, ifname, addr, mask, ipv4_type):
        """Set interface IPv4 address"""
        if not addr or not mask or (not type):
            return
        maskstr = self.convert_len_to_mask(mask)
        if self.state == 'present':
            if not self.is_ipv4_exist(addr, maskstr, ipv4_type):
                if ipv4_type == 'main':
                    main_addr = self.get_ipv4_main_addr()
                    if not main_addr:
                        xml_str = CE_NC_ADD_IPV4 % (ifname, addr, maskstr, ipv4_type)
                        self.netconf_set_config(xml_str, 'ADD_IPV4_ADDR')
                    else:
                        xml_str = CE_NC_MERGE_IPV4 % (ifname, main_addr['ifIpAddr'], main_addr['subnetMask'], addr, maskstr)
                        self.netconf_set_config(xml_str, 'MERGE_IPV4_ADDR')
                else:
                    xml_str = CE_NC_ADD_IPV4 % (ifname, addr, maskstr, ipv4_type)
                    self.netconf_set_config(xml_str, 'ADD_IPV4_ADDR')
                self.updates_cmd.append('interface %s' % ifname)
                if ipv4_type == 'main':
                    self.updates_cmd.append('ip address %s %s' % (addr, maskstr))
                else:
                    self.updates_cmd.append('ip address %s %s sub' % (addr, maskstr))
                self.changed = True
        elif self.is_ipv4_exist(addr, maskstr, ipv4_type):
            xml_str = CE_NC_DEL_IPV4 % (ifname, addr, maskstr, ipv4_type)
            self.netconf_set_config(xml_str, 'DEL_IPV4_ADDR')
            self.updates_cmd.append('interface %s' % ifname)
            if ipv4_type == 'main':
                self.updates_cmd.append('undo ip address %s %s' % (addr, maskstr))
            else:
                self.updates_cmd.append('undo ip address %s %s sub' % (addr, maskstr))
            self.changed = True

    def set_ipv6_addr(self, ifname, addr, mask):
        """Set interface IPv6 address"""
        if not addr or not mask:
            return
        if self.state == 'present':
            self.updates_cmd.append('interface %s' % ifname)
            if self.intf_info['enableFlag'] == 'false':
                xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'true')
                self.netconf_set_config(xml_str, 'SET_IPV6_ENABLE')
                self.updates_cmd.append('ipv6 enable')
                self.changed = True
            if not self.is_ipv6_exist(addr, mask):
                xml_str = CE_NC_ADD_IPV6 % (ifname, addr, mask)
                self.netconf_set_config(xml_str, 'ADD_IPV6_ADDR')
                self.updates_cmd.append('ipv6 address %s %s' % (addr, mask))
                self.changed = True
            if not self.changed:
                self.updates_cmd.pop()
        elif self.is_ipv6_exist(addr, mask):
            xml_str = CE_NC_DEL_IPV6 % (ifname, addr, mask)
            self.netconf_set_config(xml_str, 'DEL_IPV6_ADDR')
            self.updates_cmd.append('interface %s' % ifname)
            self.updates_cmd.append('undo ipv6 address %s %s' % (addr, mask))
            self.changed = True

    def set_ipv6_enable(self, ifname):
        """Set interface IPv6 enable"""
        if self.state == 'present':
            if self.intf_info['enableFlag'] == 'false':
                xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'true')
                self.netconf_set_config(xml_str, 'SET_IPV6_ENABLE')
                self.updates_cmd.append('interface %s' % ifname)
                self.updates_cmd.append('ipv6 enable')
                self.changed = True
        elif self.intf_info['enableFlag'] == 'true':
            xml_str = CE_NC_MERGE_IPV6_ENABLE % (ifname, 'false')
            self.netconf_set_config(xml_str, 'SET_IPV6_DISABLE')
            self.updates_cmd.append('interface %s' % ifname)
            self.updates_cmd.append('undo ipv6 enable')
            self.changed = True

    def check_params(self):
        """Check all input params"""
        if self.interface:
            self.intf_type = get_interface_type(self.interface)
            if not self.intf_type:
                self.module.fail_json(msg='Error: Interface name of %s is error.' % self.interface)
        if self.version == 'v4':
            if not is_valid_v4addr(self.addr):
                self.module.fail_json(msg='Error: The %s is not a valid address.' % self.addr)
            if not self.mask.isdigit():
                self.module.fail_json(msg='Error: mask is invalid.')
            if int(self.mask) > 32 or int(self.mask) < 1:
                self.module.fail_json(msg='Error: mask must be an integer between 1 and 32.')
        if self.version == 'v6':
            if self.addr:
                if not self.mask.isdigit():
                    self.module.fail_json(msg='Error: mask is invalid.')
                if int(self.mask) > 128 or int(self.mask) < 1:
                    self.module.fail_json(msg='Error: mask must be an integer between 1 and 128.')
        self.intf_info = self.get_interface_dict(self.interface)
        if not self.intf_info:
            self.module.fail_json(msg='Error: interface %s does not exist.' % self.interface)
        if self.intf_info['isL2SwitchPort'] == 'true':
            self.module.fail_json(msg='Error: interface %s is layer2.' % self.interface)

    def get_proposed(self):
        """get proposed info"""
        self.proposed['state'] = self.state
        self.proposed['addr'] = self.addr
        self.proposed['mask'] = self.mask
        self.proposed['ipv4_type'] = self.ipv4_type
        self.proposed['version'] = self.version
        self.proposed['interface'] = self.interface

    def get_existing(self):
        """get existing info"""
        self.existing['interface'] = self.interface
        self.existing['ipv4addr'] = self.intf_info['am4CfgAddr']
        self.existing['ipv6addr'] = self.intf_info['am6CfgAddr']
        self.existing['ipv6enalbe'] = self.intf_info['enableFlag']

    def get_end_state(self):
        """get end state info"""
        intf_info = self.get_interface_dict(self.interface)
        self.end_state['interface'] = self.interface
        self.end_state['ipv4addr'] = intf_info['am4CfgAddr']
        self.end_state['ipv6addr'] = intf_info['am6CfgAddr']
        self.end_state['ipv6enalbe'] = intf_info['enableFlag']

    def work(self):
        """worker"""
        self.check_params()
        self.get_existing()
        self.get_proposed()
        if self.version == 'v4':
            self.set_ipv4_addr(self.interface, self.addr, self.mask, self.ipv4_type)
        elif not self.addr and (not self.mask):
            self.set_ipv6_enable(self.interface)
        else:
            self.set_ipv6_addr(self.interface, self.addr, self.mask)
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