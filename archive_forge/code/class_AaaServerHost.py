from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec, check_ip_addr
class AaaServerHost(object):
    """ Manages aaa server host configuration """

    def netconf_get_config(self, **kwargs):
        """ Get configure by netconf """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        xml_str = get_nc_config(module, conf_str)
        return xml_str

    def netconf_set_config(self, **kwargs):
        """ Set configure by netconf """
        module = kwargs['module']
        conf_str = kwargs['conf_str']
        recv_xml = set_nc_config(module, conf_str)
        return recv_xml

    def get_local_user_info(self, **kwargs):
        """ Get local user information """
        module = kwargs['module']
        local_user_name = module.params['local_user_name']
        local_service_type = module.params['local_service_type']
        local_ftp_dir = module.params['local_ftp_dir']
        local_user_level = module.params['local_user_level']
        local_user_group = module.params['local_user_group']
        state = module.params['state']
        result = dict()
        result['local_user_info'] = []
        need_cfg = False
        conf_str = CE_GET_LOCAL_USER_INFO_HEADER
        if local_service_type:
            if local_service_type == 'none':
                conf_str += '<serviceTerminal></serviceTerminal>'
                conf_str += '<serviceTelnet></serviceTelnet>'
                conf_str += '<serviceFtp></serviceFtp>'
                conf_str += '<serviceSsh></serviceSsh>'
                conf_str += '<serviceSnmp></serviceSnmp>'
                conf_str += '<serviceDot1x></serviceDot1x>'
            elif local_service_type == 'dot1x':
                conf_str += '<serviceDot1x></serviceDot1x>'
            else:
                option = local_service_type.split(' ')
                for tmp in option:
                    if tmp == 'dot1x':
                        module.fail_json(msg='Error: Do not input dot1x with other service type.')
                    elif tmp == 'none':
                        module.fail_json(msg='Error: Do not input none with other service type.')
                    elif tmp == 'ftp':
                        conf_str += '<serviceFtp></serviceFtp>'
                    elif tmp == 'snmp':
                        conf_str += '<serviceSnmp></serviceSnmp>'
                    elif tmp == 'ssh':
                        conf_str += '<serviceSsh></serviceSsh>'
                    elif tmp == 'telnet':
                        conf_str += '<serviceTelnet></serviceTelnet>'
                    elif tmp == 'terminal':
                        conf_str += '<serviceTerminal></serviceTerminal>'
                    else:
                        module.fail_json(msg='Error: Do not support the type [%s].' % tmp)
        if local_ftp_dir:
            conf_str += '<ftpDir></ftpDir>'
        if local_user_level:
            conf_str += '<userLevel></userLevel>'
        if local_user_group:
            conf_str += '<userGroupName></userGroupName>'
        conf_str += CE_GET_LOCAL_USER_INFO_TAIL
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            local_user_info = root.findall('aaa/lam/users/user')
            if local_user_info:
                for tmp in local_user_info:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['userName', 'password', 'userLevel', 'ftpDir', 'userGroupName', 'serviceTerminal', 'serviceTelnet', 'serviceFtp', 'serviceSsh', 'serviceSnmp', 'serviceDot1x']:
                            tmp_dict[site.tag] = site.text
                    result['local_user_info'].append(tmp_dict)
            if state == 'present':
                need_cfg = True
            elif result['local_user_info']:
                for tmp in result['local_user_info']:
                    if 'userName' in tmp.keys():
                        if tmp['userName'] == local_user_name:
                            if not local_service_type and (not local_user_level) and (not local_ftp_dir) and (not local_user_group):
                                need_cfg = True
                            if local_service_type:
                                if local_service_type == 'none':
                                    if tmp.get('serviceTerminal') == 'true' or tmp.get('serviceTelnet') == 'true' or tmp.get('serviceFtp') == 'true' or (tmp.get('serviceSsh') == 'true') or (tmp.get('serviceSnmp') == 'true') or (tmp.get('serviceDot1x') == 'true'):
                                        need_cfg = True
                                elif local_service_type == 'dot1x':
                                    if tmp.get('serviceDot1x') == 'true':
                                        need_cfg = True
                                elif tmp == 'ftp':
                                    if tmp.get('serviceFtp') == 'true':
                                        need_cfg = True
                                elif tmp == 'snmp':
                                    if tmp.get('serviceSnmp') == 'true':
                                        need_cfg = True
                                elif tmp == 'ssh':
                                    if tmp.get('serviceSsh') == 'true':
                                        need_cfg = True
                                elif tmp == 'telnet':
                                    if tmp.get('serviceTelnet') == 'true':
                                        need_cfg = True
                                elif tmp == 'terminal':
                                    if tmp.get('serviceTerminal') == 'true':
                                        need_cfg = True
                            if local_user_level:
                                if tmp.get('userLevel') == local_user_level:
                                    need_cfg = True
                            if local_ftp_dir:
                                if tmp.get('ftpDir') == local_ftp_dir:
                                    need_cfg = True
                            if local_user_group:
                                if tmp.get('userGroupName') == local_user_group:
                                    need_cfg = True
                            break
        result['need_cfg'] = need_cfg
        return result

    def merge_local_user_info(self, **kwargs):
        """ Merge local user information by netconf """
        module = kwargs['module']
        local_user_name = module.params['local_user_name']
        local_password = module.params['local_password']
        local_service_type = module.params['local_service_type']
        local_ftp_dir = module.params['local_ftp_dir']
        local_user_level = module.params['local_user_level']
        local_user_group = module.params['local_user_group']
        state = module.params['state']
        cmds = []
        conf_str = CE_MERGE_LOCAL_USER_INFO_HEADER % local_user_name
        if local_password:
            conf_str += '<password>%s</password>' % local_password
        if state == 'present':
            cmd = 'local-user %s password cipher %s' % (local_user_name, local_password)
            cmds.append(cmd)
        if local_service_type:
            if local_service_type == 'none':
                conf_str += '<serviceTerminal>false</serviceTerminal>'
                conf_str += '<serviceTelnet>false</serviceTelnet>'
                conf_str += '<serviceFtp>false</serviceFtp>'
                conf_str += '<serviceSsh>false</serviceSsh>'
                conf_str += '<serviceSnmp>false</serviceSnmp>'
                conf_str += '<serviceDot1x>false</serviceDot1x>'
                cmd = 'local-user %s service-type none' % local_user_name
                cmds.append(cmd)
            elif local_service_type == 'dot1x':
                if state == 'present':
                    conf_str += '<serviceDot1x>true</serviceDot1x>'
                    cmd = 'local-user %s service-type dot1x' % local_user_name
                else:
                    conf_str += '<serviceDot1x>false</serviceDot1x>'
                    cmd = 'undo local-user %s service-type' % local_user_name
                cmds.append(cmd)
            else:
                option = local_service_type.split(' ')
                for tmp in option:
                    if tmp == 'dot1x':
                        module.fail_json(msg='Error: Do not input dot1x with other service type.')
                    if tmp == 'none':
                        module.fail_json(msg='Error: Do not input none with other service type.')
                    if state == 'present':
                        if tmp == 'ftp':
                            conf_str += '<serviceFtp>true</serviceFtp>'
                            cmd = 'local-user %s service-type ftp' % local_user_name
                        elif tmp == 'snmp':
                            conf_str += '<serviceSnmp>true</serviceSnmp>'
                            cmd = 'local-user %s service-type snmp' % local_user_name
                        elif tmp == 'ssh':
                            conf_str += '<serviceSsh>true</serviceSsh>'
                            cmd = 'local-user %s service-type ssh' % local_user_name
                        elif tmp == 'telnet':
                            conf_str += '<serviceTelnet>true</serviceTelnet>'
                            cmd = 'local-user %s service-type telnet' % local_user_name
                        elif tmp == 'terminal':
                            conf_str += '<serviceTerminal>true</serviceTerminal>'
                            cmd = 'local-user %s service-type terminal' % local_user_name
                        cmds.append(cmd)
                    elif tmp == 'ftp':
                        conf_str += '<serviceFtp>false</serviceFtp>'
                    elif tmp == 'snmp':
                        conf_str += '<serviceSnmp>false</serviceSnmp>'
                    elif tmp == 'ssh':
                        conf_str += '<serviceSsh>false</serviceSsh>'
                    elif tmp == 'telnet':
                        conf_str += '<serviceTelnet>false</serviceTelnet>'
                    elif tmp == 'terminal':
                        conf_str += '<serviceTerminal>false</serviceTerminal>'
                if state == 'absent':
                    cmd = 'undo local-user %s service-type' % local_user_name
                    cmds.append(cmd)
        if local_ftp_dir:
            if state == 'present':
                conf_str += '<ftpDir>%s</ftpDir>' % local_ftp_dir
                cmd = 'local-user %s ftp-directory %s' % (local_user_name, local_ftp_dir)
                cmds.append(cmd)
            else:
                conf_str += '<ftpDir></ftpDir>'
                cmd = 'undo local-user %s ftp-directory' % local_user_name
                cmds.append(cmd)
        if local_user_level:
            if state == 'present':
                conf_str += '<userLevel>%s</userLevel>' % local_user_level
                cmd = 'local-user %s level %s' % (local_user_name, local_user_level)
                cmds.append(cmd)
            else:
                conf_str += '<userLevel></userLevel>'
                cmd = 'undo local-user %s level' % local_user_name
                cmds.append(cmd)
        if local_user_group:
            if state == 'present':
                conf_str += '<userGroupName>%s</userGroupName>' % local_user_group
                cmd = 'local-user %s user-group %s' % (local_user_name, local_user_group)
                cmds.append(cmd)
            else:
                conf_str += '<userGroupName></userGroupName>'
                cmd = 'undo local-user %s user-group' % local_user_name
                cmds.append(cmd)
        conf_str += CE_MERGE_LOCAL_USER_INFO_TAIL
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge local user info failed.')
        return cmds

    def delete_local_user_info(self, **kwargs):
        """ Delete local user information by netconf """
        module = kwargs['module']
        local_user_name = module.params['local_user_name']
        conf_str = CE_DELETE_LOCAL_USER_INFO_HEADER % local_user_name
        conf_str += CE_DELETE_LOCAL_USER_INFO_TAIL
        cmds = []
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete local user info failed.')
        cmd = 'undo local-user %s' % local_user_name
        cmds.append(cmd)
        return cmds

    def get_radius_server_cfg_ipv4(self, **kwargs):
        """ Get radius server configure ipv4 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ip = module.params['radius_server_ip']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        state = module.params['state']
        result = dict()
        result['radius_server_ip_v4'] = []
        need_cfg = False
        conf_str = CE_GET_RADIUS_SERVER_CFG_IPV4 % radius_group_name
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            radius_server_ip_v4 = root.findall('radius/rdsTemplates/rdsTemplate/rdsServerIPV4s/rdsServerIPV4')
            if radius_server_ip_v4:
                for tmp in radius_server_ip_v4:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverType', 'serverIPAddress', 'serverPort', 'serverMode', 'vpnName']:
                            tmp_dict[site.tag] = site.text
                    result['radius_server_ip_v4'].append(tmp_dict)
            if result['radius_server_ip_v4']:
                cfg = dict()
                config_list = list()
                if radius_server_type:
                    cfg['serverType'] = radius_server_type.lower()
                if radius_server_ip:
                    cfg['serverIPAddress'] = radius_server_ip.lower()
                if radius_server_port:
                    cfg['serverPort'] = radius_server_port.lower()
                if radius_server_mode:
                    cfg['serverMode'] = radius_server_mode.lower()
                if radius_vpn_name:
                    cfg['vpnName'] = radius_vpn_name.lower()
                for tmp in result['radius_server_ip_v4']:
                    exist_cfg = dict()
                    if radius_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if radius_server_ip:
                        exist_cfg['serverIPAddress'] = tmp.get('serverIPAddress').lower()
                    if radius_server_port:
                        exist_cfg['serverPort'] = tmp.get('serverPort').lower()
                    if radius_server_mode:
                        exist_cfg['serverMode'] = tmp.get('serverMode').lower()
                    if radius_vpn_name:
                        exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_radius_server_cfg_ipv4(self, **kwargs):
        """ Merge radius server configure ipv4 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ip = module.params['radius_server_ip']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        conf_str = CE_MERGE_RADIUS_SERVER_CFG_IPV4 % (radius_group_name, radius_server_type, radius_server_ip, radius_server_port, radius_server_mode, radius_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge radius server config ipv4 failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'radius server authentication %s %s' % (radius_server_ip, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'radius server accounting %s %s' % (radius_server_ip, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_radius_server_cfg_ipv4(self, **kwargs):
        """ Delete radius server configure ipv4 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ip = module.params['radius_server_ip']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        conf_str = CE_DELETE_RADIUS_SERVER_CFG_IPV4 % (radius_group_name, radius_server_type, radius_server_ip, radius_server_port, radius_server_mode, radius_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Create radius server config ipv4 failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'undo radius server authentication %s %s' % (radius_server_ip, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'undo radius server accounting  %s %s' % (radius_server_ip, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def get_radius_server_cfg_ipv6(self, **kwargs):
        """ Get radius server configure ipv6 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ipv6 = module.params['radius_server_ipv6']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        state = module.params['state']
        result = dict()
        result['radius_server_ip_v6'] = []
        need_cfg = False
        conf_str = CE_GET_RADIUS_SERVER_CFG_IPV6 % radius_group_name
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            radius_server_ip_v6 = root.findall('radius/rdsTemplates/rdsTemplate/rdsServerIPV6s/rdsServerIPV6')
            if radius_server_ip_v6:
                for tmp in radius_server_ip_v6:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverType', 'serverIPAddress', 'serverPort', 'serverMode']:
                            tmp_dict[site.tag] = site.text
                    result['radius_server_ip_v6'].append(tmp_dict)
            if result['radius_server_ip_v6']:
                cfg = dict()
                config_list = list()
                if radius_server_type:
                    cfg['serverType'] = radius_server_type.lower()
                if radius_server_ipv6:
                    cfg['serverIPAddress'] = radius_server_ipv6.lower()
                if radius_server_port:
                    cfg['serverPort'] = radius_server_port.lower()
                if radius_server_mode:
                    cfg['serverMode'] = radius_server_mode.lower()
                for tmp in result['radius_server_ip_v6']:
                    exist_cfg = dict()
                    if radius_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if radius_server_ipv6:
                        exist_cfg['serverIPAddress'] = tmp.get('serverIPAddress').lower()
                    if radius_server_port:
                        exist_cfg['serverPort'] = tmp.get('serverPort').lower()
                    if radius_server_mode:
                        exist_cfg['serverMode'] = tmp.get('serverMode').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_radius_server_cfg_ipv6(self, **kwargs):
        """ Merge radius server configure ipv6 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ipv6 = module.params['radius_server_ipv6']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        conf_str = CE_MERGE_RADIUS_SERVER_CFG_IPV6 % (radius_group_name, radius_server_type, radius_server_ipv6, radius_server_port, radius_server_mode)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge radius server config ipv6 failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'radius server authentication %s %s' % (radius_server_ipv6, radius_server_port)
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'radius server accounting  %s %s' % (radius_server_ipv6, radius_server_port)
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_radius_server_cfg_ipv6(self, **kwargs):
        """ Delete radius server configure ipv6 """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_ipv6 = module.params['radius_server_ipv6']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        conf_str = CE_DELETE_RADIUS_SERVER_CFG_IPV6 % (radius_group_name, radius_server_type, radius_server_ipv6, radius_server_port, radius_server_mode)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Create radius server config ipv6 failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'undo radius server authentication %s %s' % (radius_server_ipv6, radius_server_port)
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'undo radius server accounting  %s %s' % (radius_server_ipv6, radius_server_port)
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def get_radius_server_name(self, **kwargs):
        """ Get radius server name """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_name = module.params['radius_server_name']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        state = module.params['state']
        result = dict()
        result['radius_server_name_cfg'] = []
        need_cfg = False
        conf_str = CE_GET_RADIUS_SERVER_NAME % radius_group_name
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            radius_server_name_cfg = root.findall('radius/rdsTemplates/rdsTemplate/rdsServerNames/rdsServerName')
            if radius_server_name_cfg:
                for tmp in radius_server_name_cfg:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverType', 'serverName', 'serverPort', 'serverMode', 'vpnName']:
                            tmp_dict[site.tag] = site.text
                    result['radius_server_name_cfg'].append(tmp_dict)
            if result['radius_server_name_cfg']:
                cfg = dict()
                config_list = list()
                if radius_server_type:
                    cfg['serverType'] = radius_server_type.lower()
                if radius_server_name:
                    cfg['serverName'] = radius_server_name.lower()
                if radius_server_port:
                    cfg['serverPort'] = radius_server_port.lower()
                if radius_server_mode:
                    cfg['serverMode'] = radius_server_mode.lower()
                if radius_vpn_name:
                    cfg['vpnName'] = radius_vpn_name.lower()
                for tmp in result['radius_server_name_cfg']:
                    exist_cfg = dict()
                    if radius_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if radius_server_name:
                        exist_cfg['serverName'] = tmp.get('serverName').lower()
                    if radius_server_port:
                        exist_cfg['serverPort'] = tmp.get('serverPort').lower()
                    if radius_server_mode:
                        exist_cfg['serverMode'] = tmp.get('serverMode').lower()
                    if radius_vpn_name:
                        exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_radius_server_name(self, **kwargs):
        """ Merge radius server name """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_name = module.params['radius_server_name']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        conf_str = CE_MERGE_RADIUS_SERVER_NAME % (radius_group_name, radius_server_type, radius_server_name, radius_server_port, radius_server_mode, radius_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge radius server name failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'radius server authentication hostname %s %s' % (radius_server_name, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'radius server accounting hostname %s %s' % (radius_server_name, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_radius_server_name(self, **kwargs):
        """ Delete radius server name """
        module = kwargs['module']
        radius_group_name = module.params['radius_group_name']
        radius_server_type = module.params['radius_server_type']
        radius_server_name = module.params['radius_server_name']
        radius_server_port = module.params['radius_server_port']
        radius_server_mode = module.params['radius_server_mode']
        radius_vpn_name = module.params['radius_vpn_name']
        conf_str = CE_DELETE_RADIUS_SERVER_NAME % (radius_group_name, radius_server_type, radius_server_name, radius_server_port, radius_server_mode, radius_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: delete radius server name failed.')
        cmds = []
        cmd = 'radius server group %s' % radius_group_name
        cmds.append(cmd)
        if radius_server_type == 'Authentication':
            cmd = 'undo radius server authentication hostname %s %s' % (radius_server_name, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        else:
            cmd = 'undo radius server accounting hostname %s %s' % (radius_server_name, radius_server_port)
            if radius_vpn_name and radius_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % radius_vpn_name
            if radius_server_mode == 'Secondary-server':
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def get_hwtacacs_server_cfg_ipv4(self, **kwargs):
        """ Get hwtacacs server configure ipv4 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ip = module.params['hwtacacs_server_ip']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
        state = module.params['state']
        result = dict()
        result['hwtacacs_server_cfg_ipv4'] = []
        need_cfg = False
        conf_str = CE_GET_HWTACACS_SERVER_CFG_IPV4 % hwtacacs_template
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            hwtacacs_server_cfg_ipv4 = root.findall('hwtacacs/hwTacTempCfgs/hwTacTempCfg/hwTacSrvCfgs/hwTacSrvCfg')
            if hwtacacs_server_cfg_ipv4:
                for tmp in hwtacacs_server_cfg_ipv4:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverIpAddress', 'serverType', 'isSecondaryServer', 'isPublicNet', 'vpnName']:
                            tmp_dict[site.tag] = site.text
                    result['hwtacacs_server_cfg_ipv4'].append(tmp_dict)
            if result['hwtacacs_server_cfg_ipv4']:
                cfg = dict()
                config_list = list()
                if hwtacacs_server_ip:
                    cfg['serverIpAddress'] = hwtacacs_server_ip.lower()
                if hwtacacs_server_type:
                    cfg['serverType'] = hwtacacs_server_type.lower()
                if hwtacacs_is_secondary_server:
                    cfg['isSecondaryServer'] = str(hwtacacs_is_secondary_server).lower()
                if hwtacacs_is_public_net:
                    cfg['isPublicNet'] = str(hwtacacs_is_public_net).lower()
                if hwtacacs_vpn_name:
                    cfg['vpnName'] = hwtacacs_vpn_name.lower()
                for tmp in result['hwtacacs_server_cfg_ipv4']:
                    exist_cfg = dict()
                    if hwtacacs_server_ip:
                        exist_cfg['serverIpAddress'] = tmp.get('serverIpAddress').lower()
                    if hwtacacs_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if hwtacacs_is_secondary_server:
                        exist_cfg['isSecondaryServer'] = tmp.get('isSecondaryServer').lower()
                    if hwtacacs_is_public_net:
                        exist_cfg['isPublicNet'] = tmp.get('isPublicNet').lower()
                    if hwtacacs_vpn_name:
                        exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_hwtacacs_server_cfg_ipv4(self, **kwargs):
        """ Merge hwtacacs server configure ipv4 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ip = module.params['hwtacacs_server_ip']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
        conf_str = CE_MERGE_HWTACACS_SERVER_CFG_IPV4 % (hwtacacs_template, hwtacacs_server_ip, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name, str(hwtacacs_is_public_net).lower())
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge hwtacacs server config ipv4 failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacacs_template
        cmds.append(cmd)
        if hwtacacs_server_type == 'Authentication':
            cmd = 'hwtacacs server authentication %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'hwtacacs server authorization %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'hwtacacs server accounting %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'hwtacacs server %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_hwtacacs_server_cfg_ipv4(self, **kwargs):
        """ Delete hwtacacs server configure ipv4 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ip = module.params['hwtacacs_server_ip']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
        conf_str = CE_DELETE_HWTACACS_SERVER_CFG_IPV4 % (hwtacacs_template, hwtacacs_server_ip, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name, str(hwtacacs_is_public_net).lower())
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete hwtacacs server config ipv4 failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacacs_template
        cmds.append(cmd)
        if hwtacacs_server_type == 'Authentication':
            cmd = 'undo hwtacacs server authentication %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'undo hwtacacs server authorization %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'undo hwtacacs server accounting %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'undo hwtacacs server %s' % hwtacacs_server_ip
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def get_hwtacacs_server_cfg_ipv6(self, **kwargs):
        """ Get hwtacacs server configure ipv6 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ipv6 = module.params['hwtacacs_server_ipv6']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        state = module.params['state']
        result = dict()
        result['hwtacacs_server_cfg_ipv6'] = []
        need_cfg = False
        conf_str = CE_GET_HWTACACS_SERVER_CFG_IPV6 % hwtacacs_template
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            hwtacacs_server_cfg_ipv6 = root.findall('hwtacacs/hwTacTempCfgs/hwTacTempCfg/hwTacIpv6SrvCfgs/hwTacIpv6SrvCfg')
            if hwtacacs_server_cfg_ipv6:
                for tmp in hwtacacs_server_cfg_ipv6:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverIpAddress', 'serverType', 'isSecondaryServer', 'vpnName']:
                            tmp_dict[site.tag] = site.text
                    result['hwtacacs_server_cfg_ipv6'].append(tmp_dict)
            if result['hwtacacs_server_cfg_ipv6']:
                cfg = dict()
                config_list = list()
                if hwtacacs_server_ipv6:
                    cfg['serverIpAddress'] = hwtacacs_server_ipv6.lower()
                if hwtacacs_server_type:
                    cfg['serverType'] = hwtacacs_server_type.lower()
                if hwtacacs_is_secondary_server:
                    cfg['isSecondaryServer'] = str(hwtacacs_is_secondary_server).lower()
                if hwtacacs_vpn_name:
                    cfg['vpnName'] = hwtacacs_vpn_name.lower()
                for tmp in result['hwtacacs_server_cfg_ipv6']:
                    exist_cfg = dict()
                    if hwtacacs_server_ipv6:
                        exist_cfg['serverIpAddress'] = tmp.get('serverIpAddress').lower()
                    if hwtacacs_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if hwtacacs_is_secondary_server:
                        exist_cfg['isSecondaryServer'] = tmp.get('isSecondaryServer').lower()
                    if hwtacacs_vpn_name:
                        exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_hwtacacs_server_cfg_ipv6(self, **kwargs):
        """ Merge hwtacacs server configure ipv6 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ipv6 = module.params['hwtacacs_server_ipv6']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        conf_str = CE_MERGE_HWTACACS_SERVER_CFG_IPV6 % (hwtacacs_template, hwtacacs_server_ipv6, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge hwtacacs server config ipv6 failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacacs_template
        cmds.append(cmd)
        if hwtacacs_server_type == 'Authentication':
            cmd = 'hwtacacs server authentication %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'hwtacacs server authorization %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'hwtacacs server accounting %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'hwtacacs server %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_hwtacacs_server_cfg_ipv6(self, **kwargs):
        """ Delete hwtacacs server configure ipv6 """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_ipv6 = module.params['hwtacacs_server_ipv6']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        conf_str = CE_DELETE_HWTACACS_SERVER_CFG_IPV6 % (hwtacacs_template, hwtacacs_server_ipv6, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name)
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete hwtacacs server config ipv6 failed.')
        cmds = []
        cmd = 'hwtacacs server template %s' % hwtacacs_template
        cmds.append(cmd)
        if hwtacacs_server_type == 'Authentication':
            cmd = 'undo hwtacacs server authentication %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'undo hwtacacs server authorization %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'undo hwtacacs server accounting %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'undo hwtacacs server %s' % hwtacacs_server_ipv6
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def get_hwtacacs_host_server_cfg(self, **kwargs):
        """ Get hwtacacs host server configure """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_host_name = module.params['hwtacacs_server_host_name']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = 'true' if module.params['hwtacacs_is_secondary_server'] is True else 'false'
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = 'true' if module.params['hwtacacs_is_public_net'] is True else 'false'
        state = module.params['state']
        result = dict()
        result['hwtacacs_server_name_cfg'] = []
        need_cfg = False
        conf_str = CE_GET_HWTACACS_HOST_SERVER_CFG % hwtacacs_template
        recv_xml = self.netconf_get_config(module=module, conf_str=conf_str)
        if '<data/>' in recv_xml:
            if state == 'present':
                need_cfg = True
        else:
            xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
            root = ElementTree.fromstring(xml_str)
            hwtacacs_server_name_cfg = root.findall('hwtacacs/hwTacTempCfgs/hwTacTempCfg/hwTacHostSrvCfgs/hwTacHostSrvCfg')
            if hwtacacs_server_name_cfg:
                for tmp in hwtacacs_server_name_cfg:
                    tmp_dict = dict()
                    for site in tmp:
                        if site.tag in ['serverHostName', 'serverType', 'isSecondaryServer', 'isPublicNet', 'vpnName']:
                            tmp_dict[site.tag] = site.text
                    result['hwtacacs_server_name_cfg'].append(tmp_dict)
            if result['hwtacacs_server_name_cfg']:
                cfg = dict()
                config_list = list()
                if hwtacacs_server_host_name:
                    cfg['serverHostName'] = hwtacacs_server_host_name.lower()
                if hwtacacs_server_type:
                    cfg['serverType'] = hwtacacs_server_type.lower()
                if hwtacacs_is_secondary_server:
                    cfg['isSecondaryServer'] = str(hwtacacs_is_secondary_server).lower()
                if hwtacacs_is_public_net:
                    cfg['isPublicNet'] = str(hwtacacs_is_public_net).lower()
                if hwtacacs_vpn_name:
                    cfg['vpnName'] = hwtacacs_vpn_name.lower()
                for tmp in result['hwtacacs_server_name_cfg']:
                    exist_cfg = dict()
                    if hwtacacs_server_host_name:
                        exist_cfg['serverHostName'] = tmp.get('serverHostName').lower()
                    if hwtacacs_server_type:
                        exist_cfg['serverType'] = tmp.get('serverType').lower()
                    if hwtacacs_is_secondary_server:
                        exist_cfg['isSecondaryServer'] = tmp.get('isSecondaryServer').lower()
                    if hwtacacs_is_public_net:
                        exist_cfg['isPublicNet'] = tmp.get('isPublicNet').lower()
                    if hwtacacs_vpn_name:
                        exist_cfg['vpnName'] = tmp.get('vpnName').lower()
                    config_list.append(exist_cfg)
                if cfg in config_list:
                    if state == 'present':
                        need_cfg = False
                    else:
                        need_cfg = True
                elif state == 'present':
                    need_cfg = True
                else:
                    need_cfg = False
        result['need_cfg'] = need_cfg
        return result

    def merge_hwtacacs_host_server_cfg(self, **kwargs):
        """ Merge hwtacacs host server configure """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_host_name = module.params['hwtacacs_server_host_name']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
        conf_str = CE_MERGE_HWTACACS_HOST_SERVER_CFG % (hwtacacs_template, hwtacacs_server_host_name, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name, str(hwtacacs_is_public_net).lower())
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Merge hwtacacs host server config failed.')
        cmds = []
        if hwtacacs_server_type == 'Authentication':
            cmd = 'hwtacacs server authentication host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'hwtacacs server authorization host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'hwtacacs server accounting host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'hwtacacs server host host-name %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds

    def delete_hwtacacs_host_server_cfg(self, **kwargs):
        """ Delete hwtacacs host server configure """
        module = kwargs['module']
        hwtacacs_template = module.params['hwtacacs_template']
        hwtacacs_server_host_name = module.params['hwtacacs_server_host_name']
        hwtacacs_server_type = module.params['hwtacacs_server_type']
        hwtacacs_is_secondary_server = module.params['hwtacacs_is_secondary_server']
        hwtacacs_vpn_name = module.params['hwtacacs_vpn_name']
        hwtacacs_is_public_net = module.params['hwtacacs_is_public_net']
        conf_str = CE_DELETE_HWTACACS_HOST_SERVER_CFG % (hwtacacs_template, hwtacacs_server_host_name, hwtacacs_server_type, str(hwtacacs_is_secondary_server).lower(), hwtacacs_vpn_name, str(hwtacacs_is_public_net).lower())
        recv_xml = self.netconf_set_config(module=module, conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            module.fail_json(msg='Error: Delete hwtacacs host server config failed.')
        cmds = []
        if hwtacacs_server_type == 'Authentication':
            cmd = 'undo hwtacacs server authentication host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Authorization':
            cmd = 'undo hwtacacs server authorization host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Accounting':
            cmd = 'undo hwtacacs server accounting host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        elif hwtacacs_server_type == 'Common':
            cmd = 'undo hwtacacs server host %s' % hwtacacs_server_host_name
            if hwtacacs_vpn_name and hwtacacs_vpn_name != '_public_':
                cmd += ' vpn-instance %s' % hwtacacs_vpn_name
            if hwtacacs_is_public_net:
                cmd += ' public-net'
            if hwtacacs_is_secondary_server:
                cmd += ' secondary'
        cmds.append(cmd)
        return cmds