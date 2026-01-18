from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class InfoCenterTrap(object):
    """ Manages info center trap configuration """

    def __init__(self, **kwargs):
        """ Init function """
        argument_spec = kwargs['argument_spec']
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)
        self.state = self.module.params['state']
        self.trap_time_stamp = self.module.params['trap_time_stamp'] or None
        self.trap_buff_enable = self.module.params['trap_buff_enable']
        self.trap_buff_size = self.module.params['trap_buff_size'] or None
        self.module_name = self.module.params['module_name'] or None
        self.channel_id = self.module.params['channel_id'] or None
        self.trap_enable = self.module.params['trap_enable']
        self.trap_level = self.module.params['trap_level'] or None
        self.cur_global_cfg = dict()
        self.cur_source_cfg = dict()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def netconf_get_config(self, conf_str):
        """ Netconf get config """
        xml_str = get_nc_config(self.module, conf_str)
        return xml_str

    def netconf_set_config(self, conf_str):
        """ Netconf set config """
        xml_str = set_nc_config(self.module, conf_str)
        return xml_str

    def check_global_args(self):
        """ Check global args """
        need_cfg = False
        find_flag = False
        self.cur_global_cfg['global_cfg'] = []
        if self.trap_time_stamp or self.trap_buff_enable != 'no_use' or self.trap_buff_size:
            if self.trap_buff_size:
                if self.trap_buff_size.isdigit():
                    if int(self.trap_buff_size) < 0 or int(self.trap_buff_size) > 1024:
                        self.module.fail_json(msg='Error: The value of trap_buff_size is out of [0 - 1024].')
                else:
                    self.module.fail_json(msg='Error: The trap_buff_size is not digit.')
            conf_str = CE_GET_TRAP_GLOBAL_HEADER
            if self.trap_time_stamp:
                conf_str += '<trapTimeStamp></trapTimeStamp>'
            if self.trap_buff_enable != 'no_use':
                conf_str += '<icTrapBuffEn></icTrapBuffEn>'
            if self.trap_buff_size:
                conf_str += '<trapBuffSize></trapBuffSize>'
            conf_str += CE_GET_TRAP_GLOBAL_TAIL
            recv_xml = self.netconf_get_config(conf_str=conf_str)
            if '<data/>' in recv_xml:
                find_flag = False
            else:
                xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
                root = ElementTree.fromstring(xml_str)
                global_cfg = root.findall('syslog/globalParam')
                if global_cfg:
                    for tmp in global_cfg:
                        tmp_dict = dict()
                        for site in tmp:
                            if site.tag in ['trapTimeStamp', 'icTrapBuffEn', 'trapBuffSize']:
                                tmp_dict[site.tag] = site.text
                        self.cur_global_cfg['global_cfg'].append(tmp_dict)
                if self.cur_global_cfg['global_cfg']:
                    for tmp in self.cur_global_cfg['global_cfg']:
                        find_flag = True
                        if self.trap_time_stamp and tmp.get('trapTimeStamp').lower() != self.trap_time_stamp:
                            find_flag = False
                        if self.trap_buff_enable != 'no_use' and tmp.get('icTrapBuffEn') != self.trap_buff_enable:
                            find_flag = False
                        if self.trap_buff_size and tmp.get('trapBuffSize') != self.trap_buff_size:
                            find_flag = False
                        if find_flag:
                            break
                else:
                    find_flag = False
            if self.state == 'present':
                need_cfg = bool(not find_flag)
            else:
                need_cfg = bool(find_flag)
        self.cur_global_cfg['need_cfg'] = need_cfg

    def check_source_args(self):
        """ Check source args """
        need_cfg = False
        find_flag = False
        self.cur_source_cfg['source_cfg'] = list()
        if self.module_name:
            if len(self.module_name) < 1 or len(self.module_name) > 31:
                self.module.fail_json(msg='Error: The module_name is out of [1 - 31].')
            if not self.channel_id:
                self.module.fail_json(msg='Error: Please input channel_id at the same time.')
            if self.channel_id:
                if self.channel_id.isdigit():
                    if int(self.channel_id) < 0 or int(self.channel_id) > 9:
                        self.module.fail_json(msg='Error: The value of channel_id is out of [0 - 9].')
                else:
                    self.module.fail_json(msg='Error: The channel_id is not digit.')
            conf_str = CE_GET_TRAP_SOURCE_HEADER
            if self.module_name != 'default':
                conf_str += '<moduleName>%s</moduleName>' % self.module_name.upper()
            else:
                conf_str += '<moduleName>default</moduleName>'
            if self.channel_id:
                conf_str += '<icChannelId></icChannelId>'
            if self.trap_enable != 'no_use':
                conf_str += '<trapEnFlg></trapEnFlg>'
            if self.trap_level:
                conf_str += '<trapEnLevel></trapEnLevel>'
            conf_str += CE_GET_TRAP_SOURCE_TAIL
            recv_xml = self.netconf_get_config(conf_str=conf_str)
            if '<data/>' in recv_xml:
                find_flag = False
            else:
                xml_str = recv_xml.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
                root = ElementTree.fromstring(xml_str)
                source_cfg = root.findall('syslog/icSources/icSource')
                if source_cfg:
                    for tmp in source_cfg:
                        tmp_dict = dict()
                        for site in tmp:
                            if site.tag in ['moduleName', 'icChannelId', 'trapEnFlg', 'trapEnLevel']:
                                tmp_dict[site.tag] = site.text
                        self.cur_source_cfg['source_cfg'].append(tmp_dict)
                if self.cur_source_cfg['source_cfg']:
                    for tmp in self.cur_source_cfg['source_cfg']:
                        find_flag = True
                        if self.module_name and tmp.get('moduleName').lower() != self.module_name.lower():
                            find_flag = False
                        if self.channel_id and tmp.get('icChannelId') != self.channel_id:
                            find_flag = False
                        if self.trap_enable != 'no_use' and tmp.get('trapEnFlg') != self.trap_enable:
                            find_flag = False
                        if self.trap_level and tmp.get('trapEnLevel') != self.trap_level:
                            find_flag = False
                        if find_flag:
                            break
                else:
                    find_flag = False
            if self.state == 'present':
                need_cfg = bool(not find_flag)
            else:
                need_cfg = bool(find_flag)
        self.cur_source_cfg['need_cfg'] = need_cfg

    def get_proposed(self):
        """ Get proposed """
        self.proposed['state'] = self.state
        if self.trap_time_stamp:
            self.proposed['trap_time_stamp'] = self.trap_time_stamp
        if self.trap_buff_enable != 'no_use':
            self.proposed['trap_buff_enable'] = self.trap_buff_enable
        if self.trap_buff_size:
            self.proposed['trap_buff_size'] = self.trap_buff_size
        if self.module_name:
            self.proposed['module_name'] = self.module_name
        if self.channel_id:
            self.proposed['channel_id'] = self.channel_id
        if self.trap_enable != 'no_use':
            self.proposed['trap_enable'] = self.trap_enable
        if self.trap_level:
            self.proposed['trap_level'] = self.trap_level

    def get_existing(self):
        """ Get existing """
        if self.cur_global_cfg['global_cfg']:
            self.existing['global_cfg'] = self.cur_global_cfg['global_cfg']
        if self.cur_source_cfg['source_cfg']:
            self.existing['source_cfg'] = self.cur_source_cfg['source_cfg']

    def get_end_state(self):
        """ Get end state """
        self.check_global_args()
        if self.cur_global_cfg['global_cfg']:
            self.end_state['global_cfg'] = self.cur_global_cfg['global_cfg']
        self.check_source_args()
        if self.cur_source_cfg['source_cfg']:
            self.end_state['source_cfg'] = self.cur_source_cfg['source_cfg']

    def merge_trap_global(self):
        """ Merge trap global """
        conf_str = CE_MERGE_TRAP_GLOBAL_HEADER
        if self.trap_time_stamp:
            conf_str += '<trapTimeStamp>%s</trapTimeStamp>' % self.trap_time_stamp.upper()
        if self.trap_buff_enable != 'no_use':
            conf_str += '<icTrapBuffEn>%s</icTrapBuffEn>' % self.trap_buff_enable
        if self.trap_buff_size:
            conf_str += '<trapBuffSize>%s</trapBuffSize>' % self.trap_buff_size
        conf_str += CE_MERGE_TRAP_GLOBAL_TAIL
        recv_xml = self.netconf_set_config(conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge trap global failed.')
        if self.trap_time_stamp:
            cmd = 'info-center timestamp trap ' + TIME_STAMP_DICT.get(self.trap_time_stamp)
            self.updates_cmd.append(cmd)
        if self.trap_buff_enable != 'no_use':
            if self.trap_buff_enable == 'true':
                cmd = 'info-center trapbuffer'
            else:
                cmd = 'undo info-center trapbuffer'
            self.updates_cmd.append(cmd)
        if self.trap_buff_size:
            cmd = 'info-center trapbuffer size %s' % self.trap_buff_size
            self.updates_cmd.append(cmd)
        self.changed = True

    def delete_trap_global(self):
        """ Delete trap global """
        conf_str = CE_MERGE_TRAP_GLOBAL_HEADER
        if self.trap_time_stamp:
            conf_str += '<trapTimeStamp>DATE_SECOND</trapTimeStamp>'
        if self.trap_buff_enable != 'no_use':
            conf_str += '<icTrapBuffEn>false</icTrapBuffEn>'
        if self.trap_buff_size:
            conf_str += '<trapBuffSize>256</trapBuffSize>'
        conf_str += CE_MERGE_TRAP_GLOBAL_TAIL
        recv_xml = self.netconf_set_config(conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: delete trap global failed.')
        if self.trap_time_stamp:
            cmd = 'undo info-center timestamp trap'
            self.updates_cmd.append(cmd)
        if self.trap_buff_enable != 'no_use':
            cmd = 'undo info-center trapbuffer'
            self.updates_cmd.append(cmd)
        if self.trap_buff_size:
            cmd = 'undo info-center trapbuffer size'
            self.updates_cmd.append(cmd)
        self.changed = True

    def merge_trap_source(self):
        """ Merge trap source """
        conf_str = CE_MERGE_TRAP_SOURCE_HEADER
        if self.module_name:
            conf_str += '<moduleName>%s</moduleName>' % self.module_name
        if self.channel_id:
            conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
        if self.trap_enable != 'no_use':
            conf_str += '<trapEnFlg>%s</trapEnFlg>' % self.trap_enable
        if self.trap_level:
            conf_str += '<trapEnLevel>%s</trapEnLevel>' % self.trap_level
        conf_str += CE_MERGE_TRAP_SOURCE_TAIL
        recv_xml = self.netconf_set_config(conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge trap source failed.')
        cmd = 'info-center source'
        if self.module_name:
            cmd += ' %s' % self.module_name
        if self.channel_id:
            cmd += ' channel %s' % self.channel_id
        if self.trap_enable != 'no_use':
            if self.trap_enable == 'true':
                cmd += ' trap state on'
            else:
                cmd += ' trap state off'
        if self.trap_level:
            cmd += ' level %s' % self.trap_level
        self.updates_cmd.append(cmd)
        self.changed = True

    def delete_trap_source(self):
        """ Delete trap source """
        if self.trap_enable == 'no_use' and (not self.trap_level):
            conf_str = CE_DELETE_TRAP_SOURCE_HEADER
            if self.module_name:
                conf_str += '<moduleName>%s</moduleName>' % self.module_name
            if self.channel_id:
                conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
            conf_str += CE_DELETE_TRAP_SOURCE_TAIL
        else:
            conf_str = CE_MERGE_TRAP_SOURCE_HEADER
            if self.module_name:
                conf_str += '<moduleName>%s</moduleName>' % self.module_name
            if self.channel_id:
                conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
            if self.trap_enable != 'no_use':
                conf_str += '<trapEnFlg>%s</trapEnFlg>' % CHANNEL_DEFAULT_TRAP_STATE.get(self.channel_id)
            if self.trap_level:
                conf_str += '<trapEnLevel>%s</trapEnLevel>' % CHANNEL_DEFAULT_TRAP_LEVEL.get(self.channel_id)
            conf_str += CE_MERGE_TRAP_SOURCE_TAIL
        recv_xml = self.netconf_set_config(conf_str=conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Delete trap source failed.')
        cmd = 'undo info-center source'
        if self.module_name:
            cmd += ' %s' % self.module_name
        if self.channel_id:
            cmd += ' channel %s' % self.channel_id
        if self.trap_enable != 'no_use':
            cmd += ' trap state'
        if self.trap_level:
            cmd += ' level'
        self.updates_cmd.append(cmd)
        self.changed = True

    def work(self):
        """ work function """
        self.check_global_args()
        self.check_source_args()
        self.get_proposed()
        self.get_existing()
        if self.state == 'present':
            if self.cur_global_cfg['need_cfg']:
                self.merge_trap_global()
            if self.cur_source_cfg['need_cfg']:
                self.merge_trap_source()
        else:
            if self.cur_global_cfg['need_cfg']:
                self.delete_trap_global()
            if self.cur_source_cfg['need_cfg']:
                self.delete_trap_source()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        self.results['updates'] = self.updates_cmd
        self.module.exit_json(**self.results)