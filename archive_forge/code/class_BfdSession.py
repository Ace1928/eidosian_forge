from __future__ import (absolute_import, division, print_function)
import sys
import socket
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
class BfdSession(object):
    """Manages BFD Session"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.session_name = self.module.params['session_name']
        self.create_type = self.module.params['create_type']
        self.addr_type = self.module.params['addr_type']
        self.out_if_name = self.module.params['out_if_name']
        self.dest_addr = self.module.params['dest_addr']
        self.src_addr = self.module.params['src_addr']
        self.vrf_name = self.module.params['vrf_name']
        self.use_default_ip = self.module.params['use_default_ip']
        self.state = self.module.params['state']
        self.local_discr = self.module.params['local_discr']
        self.remote_discr = self.module.params['remote_discr']
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
        mutually_exclusive = [('use_default_ip', 'dest_addr')]
        self.module = AnsibleModule(argument_spec=self.spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True)

    def get_bfd_dict(self):
        """bfd config dict"""
        bfd_dict = dict()
        bfd_dict['global'] = dict()
        bfd_dict['session'] = dict()
        conf_str = CE_NC_GET_BFD % self.session_name
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return bfd_dict
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        glb = root.find('bfd/bfdSchGlobal')
        if glb:
            for attr in glb:
                bfd_dict['global'][attr.tag] = attr.text
        sess = root.find('bfd/bfdCfgSessions/bfdCfgSession')
        if sess:
            for attr in sess:
                bfd_dict['session'][attr.tag] = attr.text
        return bfd_dict

    def is_session_match(self):
        """is bfd session match"""
        if not self.bfd_dict['session'] or not self.session_name:
            return False
        session = self.bfd_dict['session']
        if self.session_name != session.get('sessName', ''):
            return False
        if self.create_type and self.create_type.upper() not in session.get('createType', '').upper():
            return False
        if self.addr_type and self.addr_type != session.get('addrType').lower():
            return False
        if self.dest_addr and self.dest_addr != session.get('destAddr'):
            return False
        if self.src_addr and self.src_addr != session.get('srcAddr'):
            return False
        if self.out_if_name:
            if not session.get('outIfName'):
                return False
            if self.out_if_name.replace(' ', '').lower() != session.get('outIfName').replace(' ', '').lower():
                return False
        if self.vrf_name and self.vrf_name != session.get('vrfName'):
            return False
        if str(self.use_default_ip).lower() != session.get('useDefaultIp'):
            return False
        if self.create_type == 'static' and self.state == 'present':
            if str(self.local_discr).lower() != session.get('localDiscr', ''):
                return False
            if str(self.remote_discr).lower() != session.get('remoteDiscr', ''):
                return False
        return True

    def config_session(self):
        """configures bfd session"""
        xml_str = ''
        cmd_list = list()
        discr = list()
        if not self.session_name:
            return xml_str
        if self.bfd_dict['global'].get('bfdEnable', 'false') != 'true':
            self.module.fail_json(msg='Error: Please enable BFD globally first.')
        xml_str = '<sessName>%s</sessName>' % self.session_name
        cmd_session = 'bfd %s' % self.session_name
        if self.state == 'present':
            if not self.bfd_dict['session']:
                if not self.dest_addr and (not self.use_default_ip):
                    self.module.fail_json(msg='Error: dest_addr or use_default_ip must be set when bfd session is creating.')
                if self.create_type == 'auto':
                    xml_str += '<createType>SESS_%s</createType>' % self.create_type.upper()
                else:
                    xml_str += '<createType>SESS_STATIC</createType>'
                xml_str += '<linkType>IP</linkType>'
                cmd_session += ' bind'
                if self.addr_type:
                    xml_str += '<addrType>%s</addrType>' % self.addr_type.upper()
                else:
                    xml_str += '<addrType>IPV4</addrType>'
                if self.dest_addr:
                    xml_str += '<destAddr>%s</destAddr>' % self.dest_addr
                    cmd_session += ' peer-%s %s' % ('ipv6' if self.addr_type == 'ipv6' else 'ip', self.dest_addr)
                if self.use_default_ip:
                    xml_str += '<useDefaultIp>%s</useDefaultIp>' % str(self.use_default_ip).lower()
                    cmd_session += ' peer-ip default-ip'
                if self.vrf_name:
                    xml_str += '<vrfName>%s</vrfName>' % self.vrf_name
                    cmd_session += ' vpn-instance %s' % self.vrf_name
                if self.out_if_name:
                    xml_str += '<outIfName>%s</outIfName>' % self.out_if_name
                    cmd_session += ' interface %s' % self.out_if_name.lower()
                if self.src_addr:
                    xml_str += '<srcAddr>%s</srcAddr>' % self.src_addr
                    cmd_session += ' source-%s %s' % ('ipv6' if self.addr_type == 'ipv6' else 'ip', self.src_addr)
                if self.create_type == 'auto':
                    cmd_session += ' auto'
                else:
                    xml_str += '<localDiscr>%s</localDiscr>' % self.local_discr
                    discr.append('discriminator local %s' % self.local_discr)
                    xml_str += '<remoteDiscr>%s</remoteDiscr>' % self.remote_discr
                    discr.append('discriminator remote %s' % self.remote_discr)
            elif not self.is_session_match():
                self.module.fail_json(msg='Error: The specified BFD configuration view has been created.')
            else:
                pass
        else:
            if not self.bfd_dict['session']:
                self.module.fail_json(msg='Error: BFD session is not exist.')
            if not self.is_session_match():
                self.module.fail_json(msg='Error: BFD session parameter is invalid.')
        if self.state == 'present':
            if xml_str.endswith('</sessName>'):
                return ''
            else:
                cmd_list.insert(0, cmd_session)
                cmd_list.extend(discr)
                self.updates_cmd.extend(cmd_list)
                return '<bfdCfgSessions><bfdCfgSession operation="merge">' + xml_str + '</bfdCfgSession></bfdCfgSessions>'
        else:
            cmd_list.append('undo ' + cmd_session)
            self.updates_cmd.extend(cmd_list)
            return '<bfdCfgSessions><bfdCfgSession operation="delete">' + xml_str + '</bfdCfgSession></bfdCfgSessions>'

    def netconf_load_config(self, xml_str):
        """load bfd config by netconf"""
        if not xml_str:
            return
        xml_cfg = '\n            <config>\n            <bfd xmlns="http://www.huawei.com/netconf/vrp" content-version="1.0" format-version="1.0">\n            %s\n            </bfd>\n            </config>' % xml_str
        set_nc_config(self.module, xml_cfg)
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if not self.session_name:
            self.module.fail_json(msg='Error: Missing required arguments: session_name.')
        if self.session_name:
            if len(self.session_name) < 1 or len(self.session_name) > 15:
                self.module.fail_json(msg='Error: Session name is invalid.')
        if self.local_discr:
            if self.local_discr < 1 or self.local_discr > 16384:
                self.module.fail_json(msg='Error: Session local_discr is not ranges from 1 to 16384.')
        if self.remote_discr:
            if self.remote_discr < 1 or self.remote_discr > 4294967295:
                self.module.fail_json(msg='Error: Session remote_discr is not ranges from 1 to 4294967295.')
        if self.state == 'present' and self.create_type == 'static':
            if not self.local_discr:
                self.module.fail_json(msg='Error: Missing required arguments: local_discr.')
            if not self.remote_discr:
                self.module.fail_json(msg='Error: Missing required arguments: remote_discr.')
        if self.out_if_name:
            if not get_interface_type(self.out_if_name):
                self.module.fail_json(msg='Error: Session out_if_name is invalid.')
        if self.dest_addr:
            if not check_ip_addr(self.dest_addr):
                self.module.fail_json(msg='Error: Session dest_addr is invalid.')
        if self.src_addr:
            if not check_ip_addr(self.src_addr):
                self.module.fail_json(msg='Error: Session src_addr is invalid.')
        if self.vrf_name:
            if not is_valid_ip_vpn(self.vrf_name):
                self.module.fail_json(msg='Error: Session vrf_name is invalid.')
            if not self.dest_addr:
                self.module.fail_json(msg='Error: vrf_name and dest_addr must set at the same time.')
        if self.use_default_ip and (not self.out_if_name):
            self.module.fail_json(msg='Error: use_default_ip and out_if_name must set at the same time.')

    def get_proposed(self):
        """get proposed info"""
        self.proposed['session_name'] = self.session_name
        self.proposed['create_type'] = self.create_type
        self.proposed['addr_type'] = self.addr_type
        self.proposed['out_if_name'] = self.out_if_name
        self.proposed['dest_addr'] = self.dest_addr
        self.proposed['src_addr'] = self.src_addr
        self.proposed['vrf_name'] = self.vrf_name
        self.proposed['use_default_ip'] = self.use_default_ip
        self.proposed['state'] = self.state
        self.proposed['local_discr'] = self.local_discr
        self.proposed['remote_discr'] = self.remote_discr

    def get_existing(self):
        """get existing info"""
        if not self.bfd_dict:
            return
        self.existing['session'] = self.bfd_dict.get('session')

    def get_end_state(self):
        """get end state info"""
        bfd_dict = self.get_bfd_dict()
        if not bfd_dict:
            return
        self.end_state['session'] = bfd_dict.get('session')
        if self.end_state == self.existing:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        self.bfd_dict = self.get_bfd_dict()
        self.get_existing()
        self.get_proposed()
        xml_str = ''
        if self.session_name:
            xml_str += self.config_session()
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