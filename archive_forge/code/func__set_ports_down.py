import traceback
import lxml.etree
import ncclient
from os_ken.base import app_manager
from os_ken.lib.netconf import constants as nc_consts
from os_ken.lib import hub
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
from os_ken.lib.of_config import constants as ofc_consts
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