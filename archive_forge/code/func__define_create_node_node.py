from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _define_create_node_node(self, group, name=None, size=None, image=None):
    """
        Defines the node before to create.

        In Abiquo, you first need to 'register' or 'define' the node in
        the API before to create it into the target hypervisor.
        """
    vm = ET.Element('virtualMachine')
    if name:
        vmname = ET.SubElement(vm, 'label')
        vmname.text = name
    attrib = {'type': self.VMTPL_MIME_TYPE, 'rel': 'virtualmachinetemplate', 'href': image.extra['url']}
    ET.SubElement(vm, 'link', attrib=attrib)
    headers = {'Accept': self.NODE_MIME_TYPE, 'Content-type': self.NODE_MIME_TYPE}
    if size:
        ram = ET.SubElement(vm, 'ram')
        ram.text = str(size.ram)
    nodes_link = group.uri + '/virtualmachines'
    vm = self.connection.request(nodes_link, data=tostring(vm), headers=headers, method='POST').object
    edit_vm = get_href(vm, 'edit')
    headers = {'Accept': self.NODE_MIME_TYPE}
    return self.connection.request(edit_vm, headers=headers).object