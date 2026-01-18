from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class InterfaceOSPF(object):
    """
    Manages configuration of an OSPF interface instance.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.interface = self.module.params['interface']
        self.process_id = self.module.params['process_id']
        self.area = self.module.params['area']
        self.cost = self.module.params['cost']
        self.hello_interval = self.module.params['hello_interval']
        self.dead_interval = self.module.params['dead_interval']
        self.silent_interface = self.module.params['silent_interface']
        self.auth_mode = self.module.params['auth_mode']
        self.auth_text_simple = self.module.params['auth_text_simple']
        self.auth_key_id = self.module.params['auth_key_id']
        self.auth_text_md5 = self.module.params['auth_text_md5']
        self.state = self.module.params['state']
        self.ospf_info = dict()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def init_module(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def netconf_set_config(self, xml_str, xml_name):
        """netconf set config"""
        rcv_xml = set_nc_config(self.module, xml_str)
        if '<ok/>' not in rcv_xml:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_area_ip(self):
        """convert integer to ip address"""
        if not self.area.isdigit():
            return self.area
        addr_int = ['0'] * 4
        addr_int[0] = str((int(self.area) & 4278190080) >> 24 & 255)
        addr_int[1] = str((int(self.area) & 16711680) >> 16 & 255)
        addr_int[2] = str((int(self.area) & 65280) >> 8 & 255)
        addr_int[3] = str(int(self.area) & 255)
        return '.'.join(addr_int)

    def get_ospf_dict(self):
        """ get one ospf attributes dict."""
        ospf_info = dict()
        conf_str = CE_NC_GET_OSPF % (self.process_id, self.get_area_ip(), self.interface)
        rcv_xml = get_nc_config(self.module, conf_str)
        if '<data/>' in rcv_xml:
            return ospf_info
        xml_str = rcv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        ospfsite = root.find('ospfv2/ospfv2comm/ospfSites/ospfSite')
        if not ospfsite:
            self.module.fail_json(msg='Error: ospf process does not exist.')
        for site in ospfsite:
            if site.tag in ['processId', 'routerId', 'vrfName']:
                ospf_info[site.tag] = site.text
        ospf_info['areaId'] = ''
        areas = root.find('ospfv2/ospfv2comm/ospfSites/ospfSite/areas/area')
        if areas:
            for area in areas:
                if area.tag == 'areaId':
                    ospf_info['areaId'] = area.text
                    break
        ospf_info['interface'] = dict()
        intf = root.find('ospfv2/ospfv2comm/ospfSites/ospfSite/areas/area/interfaces/interface')
        if intf:
            for attr in intf:
                if attr.tag in ['ifName', 'networkType', 'helloInterval', 'deadInterval', 'silentEnable', 'configCost', 'authenticationMode', 'authTextSimple', 'keyId', 'authTextMd5']:
                    ospf_info['interface'][attr.tag] = attr.text
        return ospf_info

    def set_ospf_interface(self):
        """set interface ospf enable, and set its ospf attributes"""
        xml_intf = CE_NC_XML_SET_IF_NAME % self.interface
        self.updates_cmd.append('ospf %s' % self.process_id)
        self.updates_cmd.append('area %s' % self.get_area_ip())
        if self.silent_interface:
            xml_intf += CE_NC_XML_SET_SILENT % str(self.silent_interface).lower()
            if self.silent_interface:
                self.updates_cmd.append('silent-interface %s' % self.interface)
            else:
                self.updates_cmd.append('undo silent-interface %s' % self.interface)
        self.updates_cmd.append('interface %s' % self.interface)
        self.updates_cmd.append('ospf enable %s area %s' % (self.process_id, self.get_area_ip()))
        if self.cost:
            xml_intf += CE_NC_XML_SET_COST % self.cost
            self.updates_cmd.append('ospf cost %s' % self.cost)
        if self.hello_interval:
            xml_intf += CE_NC_XML_SET_HELLO % self.hello_interval
            self.updates_cmd.append('ospf timer hello %s' % self.hello_interval)
        if self.dead_interval:
            xml_intf += CE_NC_XML_SET_DEAD % self.dead_interval
            self.updates_cmd.append('ospf timer dead %s' % self.dead_interval)
        if self.auth_mode:
            xml_intf += CE_NC_XML_SET_AUTH_MODE % self.auth_mode
            if self.auth_mode == 'none':
                self.updates_cmd.append('undo ospf authentication-mode')
            else:
                self.updates_cmd.append('ospf authentication-mode %s' % self.auth_mode)
            if self.auth_mode == 'simple' and self.auth_text_simple:
                xml_intf += CE_NC_XML_SET_AUTH_TEXT_SIMPLE % self.auth_text_simple
                self.updates_cmd.pop()
                self.updates_cmd.append('ospf authentication-mode %s %s' % (self.auth_mode, self.auth_text_simple))
            elif self.auth_mode in ['hmac-sha256', 'md5', 'hmac-md5'] and self.auth_key_id:
                xml_intf += CE_NC_XML_SET_AUTH_MD5 % (self.auth_key_id, self.auth_text_md5)
                self.updates_cmd.pop()
                self.updates_cmd.append('ospf authentication-mode %s %s %s' % (self.auth_mode, self.auth_key_id, self.auth_text_md5))
            else:
                pass
        xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, self.get_area_ip(), CE_NC_XML_BUILD_MERGE_INTF % xml_intf)
        self.netconf_set_config(xml_str, 'SET_INTERFACE_OSPF')
        self.changed = True

    def merge_ospf_interface(self):
        """merge interface ospf attributes"""
        intf_dict = self.ospf_info['interface']
        xml_ospf = ''
        if intf_dict.get('silentEnable') != str(self.silent_interface).lower():
            xml_ospf += CE_NC_XML_SET_SILENT % str(self.silent_interface).lower()
            self.updates_cmd.append('ospf %s' % self.process_id)
            self.updates_cmd.append('area %s' % self.get_area_ip())
            if self.silent_interface:
                self.updates_cmd.append('silent-interface %s' % self.interface)
            else:
                self.updates_cmd.append('undo silent-interface %s' % self.interface)
        xml_intf = ''
        self.updates_cmd.append('interface %s' % self.interface)
        if self.cost and intf_dict.get('configCost') != self.cost:
            xml_intf += CE_NC_XML_SET_COST % self.cost
            self.updates_cmd.append('ospf cost %s' % self.cost)
        if self.hello_interval and intf_dict.get('helloInterval') != self.hello_interval:
            xml_intf += CE_NC_XML_SET_HELLO % self.hello_interval
            self.updates_cmd.append('ospf timer hello %s' % self.hello_interval)
        if self.dead_interval and intf_dict.get('deadInterval') != self.dead_interval:
            xml_intf += CE_NC_XML_SET_DEAD % self.dead_interval
            self.updates_cmd.append('ospf timer dead %s' % self.dead_interval)
        if self.auth_mode:
            xml_intf += CE_NC_XML_SET_AUTH_MODE % self.auth_mode
            if self.auth_mode == 'none':
                self.updates_cmd.append('undo ospf authentication-mode')
            else:
                self.updates_cmd.append('ospf authentication-mode %s' % self.auth_mode)
            if self.auth_mode == 'simple' and self.auth_text_simple:
                xml_intf += CE_NC_XML_SET_AUTH_TEXT_SIMPLE % self.auth_text_simple
                self.updates_cmd.pop()
                self.updates_cmd.append('ospf authentication-mode %s %s' % (self.auth_mode, self.auth_text_simple))
            elif self.auth_mode in ['hmac-sha256', 'md5', 'hmac-md5'] and self.auth_key_id:
                xml_intf += CE_NC_XML_SET_AUTH_MD5 % (self.auth_key_id, self.auth_text_md5)
                self.updates_cmd.pop()
                self.updates_cmd.append('ospf authentication-mode %s %s %s' % (self.auth_mode, self.auth_key_id, self.auth_text_md5))
            else:
                pass
        if not xml_intf:
            self.updates_cmd.pop()
        if not xml_ospf and (not xml_intf):
            return
        xml_sum = CE_NC_XML_SET_IF_NAME % self.interface
        xml_sum += xml_ospf + xml_intf
        xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, self.get_area_ip(), CE_NC_XML_BUILD_MERGE_INTF % xml_sum)
        self.netconf_set_config(xml_str, 'MERGE_INTERFACE_OSPF')
        self.changed = True

    def unset_ospf_interface(self):
        """set interface ospf disable, and all its ospf attributes will be removed"""
        intf_dict = self.ospf_info['interface']
        xml_sum = ''
        xml_intf = CE_NC_XML_SET_IF_NAME % self.interface
        if intf_dict.get('silentEnable') == 'true':
            xml_sum += CE_NC_XML_BUILD_MERGE_INTF % (xml_intf + CE_NC_XML_SET_SILENT % 'false')
            self.updates_cmd.append('ospf %s' % self.process_id)
            self.updates_cmd.append('area %s' % self.get_area_ip())
            self.updates_cmd.append('undo silent-interface %s' % self.interface)
        xml_sum += CE_NC_XML_BUILD_DELETE_INTF % xml_intf
        xml_str = CE_NC_XML_BUILD_PROCESS % (self.process_id, self.get_area_ip(), xml_sum)
        self.netconf_set_config(xml_str, 'DELETE_INTERFACE_OSPF')
        self.updates_cmd.append('undo ospf cost')
        self.updates_cmd.append('undo ospf timer hello')
        self.updates_cmd.append('undo ospf timer dead')
        self.updates_cmd.append('undo ospf authentication-mode')
        self.updates_cmd.append('undo ospf enable %s area %s' % (self.process_id, self.get_area_ip()))
        self.changed = True

    def check_params(self):
        """Check all input params"""
        self.interface = self.interface.replace(' ', '').upper()
        if not get_interface_type(self.interface):
            self.module.fail_json(msg='Error: interface is invalid.')
        if not self.process_id.isdigit():
            self.module.fail_json(msg='Error: process_id is not digit.')
        if int(self.process_id) < 1 or int(self.process_id) > 4294967295:
            self.module.fail_json(msg='Error: process_id must be an integer between 1 and 4294967295.')
        if self.area.isdigit():
            if int(self.area) < 0 or int(self.area) > 4294967295:
                self.module.fail_json(msg='Error: area id (Integer) must be between 0 and 4294967295.')
        elif not is_valid_v4addr(self.area):
            self.module.fail_json(msg='Error: area id is invalid.')
        if self.state == 'present':
            if self.auth_mode:
                if self.auth_mode == 'simple':
                    if self.auth_text_simple and len(self.auth_text_simple) > 8:
                        self.module.fail_json(msg='Error: auth_text_simple is not in the range from 1 to 8.')
                if self.auth_mode in ['hmac-sha256', 'hmac-sha256', 'md5']:
                    if self.auth_key_id and (not self.auth_text_md5):
                        self.module.fail_json(msg='Error: auth_key_id and auth_text_md5 should be set at the same time.')
                    if not self.auth_key_id and self.auth_text_md5:
                        self.module.fail_json(msg='Error: auth_key_id and auth_text_md5 should be set at the same time.')
                    if self.auth_key_id:
                        if not self.auth_key_id.isdigit():
                            self.module.fail_json(msg='Error: auth_key_id is not digit.')
                        if int(self.auth_key_id) < 1 or int(self.auth_key_id) > 255:
                            self.module.fail_json(msg='Error: auth_key_id is not in the range from 1 to 255.')
                    if self.auth_text_md5 and len(self.auth_text_md5) > 255:
                        self.module.fail_json(msg='Error: auth_text_md5 is not in the range from 1 to 255.')
        if self.cost:
            if not self.cost.isdigit():
                self.module.fail_json(msg='Error: cost is not digit.')
            if int(self.cost) < 1 or int(self.cost) > 65535:
                self.module.fail_json(msg='Error: cost is not in the range from 1 to 65535')
        if self.hello_interval:
            if not self.hello_interval.isdigit():
                self.module.fail_json(msg='Error: hello_interval is not digit.')
            if int(self.hello_interval) < 1 or int(self.hello_interval) > 65535:
                self.module.fail_json(msg='Error: hello_interval is not in the range from 1 to 65535')
        if self.dead_interval:
            if not self.dead_interval.isdigit():
                self.module.fail_json(msg='Error: dead_interval is not digit.')
            if int(self.dead_interval) < 1 or int(self.dead_interval) > 235926000:
                self.module.fail_json(msg='Error: dead_interval is not in the range from 1 to 235926000')

    def get_proposed(self):
        """get proposed info"""
        self.proposed['interface'] = self.interface
        self.proposed['process_id'] = self.process_id
        self.proposed['area'] = self.get_area_ip()
        self.proposed['cost'] = self.cost
        self.proposed['hello_interval'] = self.hello_interval
        self.proposed['dead_interval'] = self.dead_interval
        self.proposed['silent_interface'] = self.silent_interface
        if self.auth_mode:
            self.proposed['auth_mode'] = self.auth_mode
            if self.auth_mode == 'simple':
                self.proposed['auth_text_simple'] = self.auth_text_simple
            if self.auth_mode in ['hmac-sha256', 'hmac-sha256', 'md5']:
                self.proposed['auth_key_id'] = self.auth_key_id
                self.proposed['auth_text_md5'] = self.auth_text_md5
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if not self.ospf_info:
            return
        if self.ospf_info['interface']:
            self.existing['interface'] = self.interface
            self.existing['cost'] = self.ospf_info['interface'].get('configCost')
            self.existing['hello_interval'] = self.ospf_info['interface'].get('helloInterval')
            self.existing['dead_interval'] = self.ospf_info['interface'].get('deadInterval')
            self.existing['silent_interface'] = self.ospf_info['interface'].get('silentEnable')
            self.existing['auth_mode'] = self.ospf_info['interface'].get('authenticationMode')
            self.existing['auth_text_simple'] = self.ospf_info['interface'].get('authTextSimple')
            self.existing['auth_key_id'] = self.ospf_info['interface'].get('keyId')
            self.existing['auth_text_md5'] = self.ospf_info['interface'].get('authTextMd5')
        self.existing['process_id'] = self.ospf_info['processId']
        self.existing['area'] = self.ospf_info['areaId']

    def get_end_state(self):
        """get end state info"""
        ospf_info = self.get_ospf_dict()
        if not ospf_info:
            return
        if ospf_info['interface']:
            self.end_state['interface'] = self.interface
            self.end_state['cost'] = ospf_info['interface'].get('configCost')
            self.end_state['hello_interval'] = ospf_info['interface'].get('helloInterval')
            self.end_state['dead_interval'] = ospf_info['interface'].get('deadInterval')
            self.end_state['silent_interface'] = ospf_info['interface'].get('silentEnable')
            self.end_state['auth_mode'] = ospf_info['interface'].get('authenticationMode')
            self.end_state['auth_text_simple'] = ospf_info['interface'].get('authTextSimple')
            self.end_state['auth_key_id'] = ospf_info['interface'].get('keyId')
            self.end_state['auth_text_md5'] = ospf_info['interface'].get('authTextMd5')
        self.end_state['process_id'] = ospf_info['processId']
        self.end_state['area'] = ospf_info['areaId']

    def work(self):
        """worker"""
        self.check_params()
        self.ospf_info = self.get_ospf_dict()
        self.get_existing()
        self.get_proposed()
        if self.state == 'present':
            if not self.ospf_info or not self.ospf_info['interface']:
                self.set_ospf_interface()
            else:
                self.merge_ospf_interface()
        elif self.ospf_info and self.ospf_info['interface']:
            self.unset_ospf_interface()
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