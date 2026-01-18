import traceback
import lxml.etree
import ncclient
from os_ken.base import app_manager
from os_ken.lib.netconf import constants as nc_consts
from os_ken.lib import hub
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
from os_ken.lib.of_config import constants as ofc_consts
class OFConfigClient(app_manager.OSKenApp):

    def __init__(self, *args, **kwargs):
        super(OFConfigClient, self).__init__(*args, **kwargs)
        self.switch = capable_switch.OFCapableSwitch(host=HOST, port=PORT, username=USERNAME, password=PASSWORD, unknown_host_cb=lambda host, fingeprint: True)
        hub.spawn(self._do_of_config)

    def _validate(self, tree):
        xmlschema = _get_schema()
        try:
            xmlschema.assertValid(tree)
        except:
            traceback.print_exc()

    def _do_get(self):
        data_xml = self.switch.raw_get()
        tree = lxml.etree.fromstring(data_xml)
        self._validate(tree)
        name_spaces = set()
        for e in tree.iter():
            name_spaces.add(capable_switch.get_ns_tag(e.tag)[0])
        print(name_spaces)
        return tree

    def _do_get_config(self, source):
        print('source = %s' % source)
        config_xml = self.switch.raw_get_config(source)
        tree = lxml.etree.fromstring(config_xml)
        self._validate(tree)

    def _do_edit_config(self, config):
        tree = lxml.etree.fromstring(config)
        self._validate(tree)
        self.switch.raw_edit_config(target='running', config=config)

    def _print_ports(self, tree, ns):
        for port in tree.findall('{%s}%s/{%s}%s' % (ns, ofc_consts.RESOURCES, ns, ofc_consts.PORT)):
            print(lxml.etree.tostring(port, pretty_print=True))

    def _set_ports_down(self):
        """try to set all ports down with etree operation"""
        tree = self._do_get()
        print(lxml.etree.tostring(tree, pretty_print=True))
        qname = lxml.etree.QName(tree.tag)
        ns = qname.namespace
        self._print_ports(tree, ns)
        switch_id = tree.find('{%s}%s' % (ns, ofc_consts.ID))
        resources = tree.find('{%s}%s' % (ns, ofc_consts.RESOURCES))
        configuration = tree.find('{%s}%s/{%s}%s/{%s}%s' % (ns, ofc_consts.RESOURCES, ns, ofc_consts.PORT, ns, ofc_consts.CONFIGURATION))
        admin_state = tree.find('{%s}%s/{%s}%s/{%s}%s/{%s}%s' % (ns, ofc_consts.RESOURCES, ns, ofc_consts.PORT, ns, ofc_consts.CONFIGURATION, ns, ofc_consts.ADMIN_STATE))
        config_ = lxml.etree.Element('{%s}%s' % (ncclient.xml_.BASE_NS_1_0, nc_consts.CONFIG))
        capable_switch_ = lxml.etree.SubElement(config_, tree.tag)
        switch_id_ = lxml.etree.SubElement(capable_switch_, switch_id.tag)
        switch_id_.text = switch_id.text
        resources_ = lxml.etree.SubElement(capable_switch_, resources.tag)
        for port in tree.findall('{%s}%s/{%s}%s' % (ns, ofc_consts.RESOURCES, ns, ofc_consts.PORT)):
            resource_id = port.find('{%s}%s' % (ns, ofc_consts.RESOURCE_ID))
            port_ = lxml.etree.SubElement(resources_, port.tag)
            resource_id_ = lxml.etree.SubElement(port_, resource_id.tag)
            resource_id_.text = resource_id.text
            configuration_ = lxml.etree.SubElement(port_, configuration.tag)
            configuration_.set(ofc_consts.OPERATION, nc_consts.MERGE)
            admin_state_ = lxml.etree.SubElement(configuration_, admin_state.tag)
            admin_state_.text = ofc_consts.DOWN
        self._do_edit_config(lxml.etree.tostring(config_, pretty_print=True))
        tree = self._do_get()
        self._print_ports(tree, ns)

    def _do_of_config(self):
        self._do_get()
        self._do_get_config('running')
        self._do_get_config('startup')
        try:
            self._do_get_config('candidate')
        except ncclient.NCClientError:
            traceback.print_exc()
        self._do_edit_config(SWITCH_PORT_DOWN)
        self._do_edit_config(SWITCH_ADVERTISED)
        self._do_edit_config(SWITCH_CONTROLLER)
        self._set_ports_down()
        self.switch.close_session()