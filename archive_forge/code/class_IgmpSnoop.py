from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
class IgmpSnoop(object):
    """igmp snooping module"""

    def __init__(self, argument_spec):
        """igmp snooping info"""
        self.spec = argument_spec
        self.module = None
        self._initmodule_()
        self.aftype = self.module.params['aftype']
        self.state = self.module.params['state']
        if self.aftype == 'v4':
            self.addr_family = 'ipv4unicast'
        else:
            self.addr_family = 'ipv6unicast'
        self.features = self.module.params['features']
        self.vlan_id = self.module.params['vlan_id']
        self.igmp = str(self.module.params['igmp']).lower()
        self.version = self.module.params['version']
        if self.version is None:
            self.version = 2
        self.proxy = str(self.module.params['proxy']).lower()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.igmp_info_data = dict()

    def _initmodule_(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=False)

    def _checkresponse_(self, xml_str, xml_name):
        """check if response message is already succeed."""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def _checkparams_(self):
        """check all input params"""
        if self.features == 'vlan':
            if not self.vlan_id:
                self.module.fail_json(msg='Error: missing required arguments: vlan_id.')
        if self.vlan_id:
            if self.vlan_id <= 0 or self.vlan_id > 4094:
                self.module.fail_json(msg='Error: Vlan id is not in the range from 1 to 4094.')
        if self.version:
            if self.version <= 0 or self.version > 3:
                self.module.fail_json(msg='Error: Version id is not in the range from 1 to 3.')

    def set_change_state(self):
        """set change state"""
        state = self.state
        change = False
        if self.features == 'vlan':
            self.get_igmp_vlan()
            change = self.compare_data()
        else:
            self.get_igmp_global()
            if state == 'present':
                if not self.igmp_info_data['igmp_info']:
                    change = True
            elif self.igmp_info_data['igmp_info']:
                change = True
        self.changed = change

    def compare_data(self):
        """compare new data and old data"""
        state = self.state
        change = False
        if state == 'present':
            if self.igmp_info_data['igmp_info']:
                for data in self.igmp_info_data['igmp_info']:
                    if self.addr_family == data['addrFamily'] and str(self.vlan_id) == data['vlanId']:
                        if self.igmp:
                            if self.igmp != data['snoopingEnable']:
                                change = True
                        if self.version:
                            if str(self.version) != data['version']:
                                change = True
                        if self.proxy:
                            if self.proxy != data['proxyEnable']:
                                change = True
            else:
                change = True
        elif self.igmp_info_data['igmp_info']:
            change = True
        return change

    def get_igmp_vlan(self):
        """get igmp vlan info data"""
        self.igmp_info_data['igmp_info'] = list()
        getxmlstr = CE_NC_GET_IGMP_VLAN_INFO % (self.addr_family, self.vlan_id)
        xml_str = get_nc_config(self.module, getxmlstr)
        if 'data/' in xml_str:
            return
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        igmp_enable = root.findall('l2mc/vlan/l2McVlanCfgs/l2McVlanCfg')
        if igmp_enable:
            for igmp_enable_key in igmp_enable:
                igmp_global_info = dict()
                for ele in igmp_enable_key:
                    if ele.tag in ['addrFamily', 'vlanId', 'snoopingEnable', 'version', 'proxyEnable']:
                        igmp_global_info[ele.tag] = ele.text
                self.igmp_info_data['igmp_info'].append(igmp_global_info)

    def get_igmp_global(self):
        """get igmp global data"""
        self.igmp_info_data['igmp_info'] = list()
        getxmlstr = CE_NC_GET_IGMP_GLOBAL % self.addr_family
        xml_str = get_nc_config(self.module, getxmlstr)
        if 'data/' in xml_str:
            return
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        igmp_enable = root.findall('l2mc/l2McSnpgEnables/l2McSnpgEnable')
        if igmp_enable:
            for igmp_enable_key in igmp_enable:
                igmp_global_info = dict()
                for ele in igmp_enable_key:
                    if ele.tag in ['addrFamily']:
                        igmp_global_info[ele.tag] = ele.text
                        self.igmp_info_data['igmp_info'].append(igmp_global_info)

    def set_vlanview_igmp(self):
        """set igmp of vlanview"""
        if not self.changed:
            return
        addr_family = self.addr_family
        state = self.state
        igmp_xml = '\n'
        version_xml = '\n'
        proxy_xml = '\n'
        if state == 'present':
            if self.igmp:
                igmp_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_SNOENABLE, self.igmp.lower())
            if str(self.version):
                version_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_VERSION, self.version)
            if self.proxy:
                proxy_xml = get_xml(CE_NC_MERGE_IGMP_VLANVIEW_PROXYENABLE, self.proxy.lower())
            configxmlstr = CE_NC_MERGE_IGMP_VLANVIEW % (addr_family, self.vlan_id, igmp_xml, version_xml, proxy_xml)
        else:
            configxmlstr = CE_NC_DELETE_IGMP_VLANVIEW % (addr_family, self.vlan_id)
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'SET_VLANVIEW_IGMP')

    def set_sysview_igmp(self):
        """set igmp of sysview"""
        if not self.changed:
            return
        version = self.addr_family
        state = self.state
        if state == 'present':
            configxmlstr = CE_NC_MERGE_IGMP_SYSVIEW % version
        else:
            configxmlstr = CE_NC_DELETE_IGMP_SYSVIEW % version
        conf_str = build_config_xml(configxmlstr)
        recv_xml = set_nc_config(self.module, conf_str)
        self._checkresponse_(recv_xml, 'SET_SYSVIEW_IGMP')

    def set_sysview_cmd(self):
        """set sysview update command"""
        if not self.changed:
            return
        if self.state == 'present':
            self.updates_cmd.append('igmp snooping enable')
        else:
            self.updates_cmd.append('undo igmp snooping enable')

    def set_vlanview_cmd(self):
        """set vlanview update command"""
        if not self.changed:
            return
        if self.state == 'present':
            if self.igmp:
                if self.igmp.lower() == 'true':
                    self.updates_cmd.append('igmp snooping enable')
                else:
                    self.updates_cmd.append('undo igmp snooping enable')
            if str(self.version):
                self.updates_cmd.append('igmp snooping version %s' % self.version)
            else:
                self.updates_cmd.append('undo igmp snooping version')
            if self.proxy:
                if self.proxy.lower() == 'true':
                    self.updates_cmd.append('igmp snooping proxy')
                else:
                    self.updates_cmd.append('undo igmp snooping proxy')
        else:
            self.updates_cmd.append('undo igmp snooping enable')
            self.updates_cmd.append('undo igmp snooping version')
            self.updates_cmd.append('undo igmp snooping proxy')

    def get_existing(self):
        """get existing information"""
        self.set_change_state()
        self.existing['igmp_info'] = self.igmp_info_data['igmp_info']

    def get_proposed(self):
        """get proposed information"""
        self.proposed['addrFamily'] = self.addr_family
        self.proposed['features'] = self.features
        if self.features == 'vlan':
            self.proposed['snoopingEnable'] = self.igmp
            self.proposed['version'] = self.version
            self.proposed['vlanId'] = self.vlan_id
            self.proposed['proxyEnable'] = self.proxy
        self.proposed['state'] = self.state

    def set_igmp_netconf(self):
        """config netconf"""
        if self.features == 'vlan':
            self.set_vlanview_igmp()
        else:
            self.set_sysview_igmp()

    def set_update_cmd(self):
        """set update command"""
        if self.features == 'vlan':
            self.set_vlanview_cmd()
        else:
            self.set_sysview_cmd()

    def get_end_state(self):
        """get end state information"""
        if self.features == 'vlan':
            self.get_igmp_vlan()
        else:
            self.get_igmp_global()
        self.end_state['igmp_info'] = self.igmp_info_data['igmp_info']

    def work(self):
        """worker"""
        self._checkparams_()
        self.get_existing()
        self.get_proposed()
        self.set_igmp_netconf()
        self.set_update_cmd()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['existing'] = self.existing
        self.results['proposed'] = self.proposed
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)