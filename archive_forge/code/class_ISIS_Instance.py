from __future__ import absolute_import, division, print_function
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
class ISIS_Instance(object):
    """Manages ISIS Instance"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.instance_id = self.module.params['instance_id']
        self.vpn_name = self.module.params['vpn_name']
        self.state = self.module.params['state']
        self.changed = False
        self.isis_dict = dict()
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def __init_module__(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def get_isis_dict(self):
        """isis config dict"""
        isis_dict = dict()
        isis_dict['instance'] = dict()
        conf_str = CE_NC_GET_ISIS % (CE_NC_GET_ISIS_INSTANCE % self.instance_id)
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return isis_dict
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        glb = root.find('isiscomm/isSites/isSite')
        if glb:
            for attr in glb:
                isis_dict['instance'][attr.tag] = attr.text
        return isis_dict

    def config_session(self):
        """configures isis"""
        xml_str = ''
        instance = self.isis_dict['instance']
        if not self.instance_id:
            return xml_str
        if self.state == 'present':
            xml_str = '<instanceId>%s</instanceId>' % self.instance_id
            self.updates_cmd.append('isis %s' % self.instance_id)
            if self.vpn_name:
                xml_str += '<vpnName>%s</vpnName>' % self.vpn_name
                self.updates_cmd.append('vpn-instance %s' % self.vpn_name)
        elif self.instance_id and str(self.instance_id) == instance.get('instanceId'):
            xml_str = '<instanceId>%s</instanceId>' % self.instance_id
            self.updates_cmd.append('undo isis %s' % self.instance_id)
        if self.state == 'present':
            return '<isSites><isSite operation="merge">' + xml_str + '</isSite></isSites>'
        elif xml_str:
            return '<isSites><isSite operation="delete">' + xml_str + '</isSite></isSites>'

    def netconf_load_config(self, xml_str):
        """load isis config by netconf"""
        if not xml_str:
            return
        xml_cfg = '\n            <config>\n            <isiscomm xmlns="http://www.huawei.com/netconf/vrp" content-version="1.0" format-version="1.0">\n            %s\n            </isiscomm>\n            </config>' % xml_str
        set_nc_config(self.module, xml_cfg)
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if not self.instance_id:
            self.module.fail_json(msg='Error: Missing required arguments: instance_id.')
        if self.instance_id:
            if self.instance_id < 1 or self.instance_id > 4294967295:
                self.module.fail_json(msg='Error: Instance id is not ranges from 1 to 4294967295.')
        if self.vpn_name:
            if not is_valid_ip_vpn(self.vpn_name):
                self.module.fail_json(msg='Error: Session vpn_name is invalid.')

    def get_proposed(self):
        """get proposed info"""
        self.proposed['instance_id'] = self.instance_id
        self.proposed['vpn_name'] = self.vpn_name
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if not self.isis_dict:
            self.existing['instance'] = None
        self.existing['instance'] = self.isis_dict.get('instance')

    def get_end_state(self):
        """get end state info"""
        isis_dict = self.get_isis_dict()
        if not isis_dict:
            self.end_state['instance'] = None
        self.end_state['instance'] = isis_dict.get('instance')
        if self.end_state == self.existing:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        self.isis_dict = self.get_isis_dict()
        self.get_existing()
        self.get_proposed()
        xml_str = ''
        if self.instance_id:
            cfg_str = self.config_session()
            if cfg_str:
                xml_str += cfg_str
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