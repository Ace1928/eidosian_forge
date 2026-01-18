import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebula_2_0_NodeDriver(OpenNebulaNodeDriver):
    """
    OpenNebula.org node driver for OpenNebula.org v2.0 through OpenNebula.org
    v2.2.
    """
    name = 'OpenNebula (v2.0 - v2.2)'

    def create_node(self, name, size, image, networks=None, context=None):
        """
        Create a new OpenNebula node.

        @inherits: :class:`NodeDriver.create_node`

        :keyword networks: List of virtual networks to which this node should
                           connect. (optional)
        :type    networks: :class:`OpenNebulaNetwork` or ``list``
                           of :class:`OpenNebulaNetwork`

        :keyword context: Custom (key, value) pairs to be injected into
                          compute node XML description. (optional)
        :type    context: ``dict``

        :return: Instance of a newly created node.
        :rtype:  :class:`Node`
        """
        compute = ET.Element('COMPUTE')
        name = ET.SubElement(compute, 'NAME')
        name.text = name
        instance_type = ET.SubElement(compute, 'INSTANCE_TYPE')
        instance_type.text = size.name
        disk = ET.SubElement(compute, 'DISK')
        ET.SubElement(disk, 'STORAGE', {'href': '/storage/%s' % str(image.id)})
        if networks:
            if not isinstance(networks, list):
                networks = [networks]
            for network in networks:
                nic = ET.SubElement(compute, 'NIC')
                ET.SubElement(nic, 'NETWORK', {'href': '/network/%s' % str(network.id)})
                if network.address:
                    ip_line = ET.SubElement(nic, 'IP')
                    ip_line.text = network.address
        if context and isinstance(context, dict):
            contextGroup = ET.SubElement(compute, 'CONTEXT')
            for key, value in list(context.items()):
                context = ET.SubElement(contextGroup, key.upper())
                context.text = value
        xml = ET.tostring(compute)
        node = self.connection.request('/compute', method='POST', data=xml).object
        return self._to_node(node)

    def destroy_node(self, node):
        url = '/compute/%s' % str(node.id)
        resp = self.connection.request(url, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def list_sizes(self, location=None):
        """
        Return list of sizes on a provider.

        @inherits: :class:`NodeDriver.list_sizes`

        :return: List of compute node sizes supported by the cloud provider.
        :rtype:  ``list`` of :class:`OpenNebulaNodeSize`
        """
        return [OpenNebulaNodeSize(id=1, name='small', ram=1024, cpu=1, disk=None, bandwidth=None, price=None, driver=self), OpenNebulaNodeSize(id=2, name='medium', ram=4096, cpu=4, disk=None, bandwidth=None, price=None, driver=self), OpenNebulaNodeSize(id=3, name='large', ram=8192, cpu=8, disk=None, bandwidth=None, price=None, driver=self), OpenNebulaNodeSize(id=4, name='custom', ram=0, cpu=0, disk=None, bandwidth=None, price=None, driver=self)]

    def _to_images(self, object):
        """
        Request a list of images and convert that list to a list of NodeImage
        objects.

        Request a list of images from the OpenNebula web interface, and
        issue a request to convert each XML object representation of an image
        to a NodeImage object.

        :rtype:  ``list`` of :class:`NodeImage`
        :return: List of images.
        """
        images = []
        for element in object.findall('STORAGE'):
            image_id = element.attrib['href'].partition('/storage/')[2]
            image = self.connection.request('/storage/%s' % image_id).object
            images.append(self._to_image(image))
        return images

    def _to_image(self, image):
        """
        Take XML object containing an image description and convert to
        NodeImage object.

        :type  image: :class:`ElementTree`
        :param image: XML representation of an image.

        :rtype:  :class:`NodeImage`
        :return: The newly extracted :class:`NodeImage`.
        """
        return NodeImage(id=image.findtext('ID'), name=image.findtext('NAME'), driver=self.connection.driver, extra={'description': image.findtext('DESCRIPTION'), 'type': image.findtext('TYPE'), 'size': image.findtext('SIZE'), 'fstype': image.findtext('FSTYPE', None)})

    def _to_node(self, compute):
        """
        Take XML object containing a compute node description and convert to
        Node object.

        Take XML representation containing a compute node description and
        convert to Node object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  :class:`Node`
        :return: The newly extracted :class:`Node`.
        """
        try:
            state = self.NODE_STATE_MAP[compute.findtext('STATE').upper()]
        except KeyError:
            state = NodeState.UNKNOWN
        return Node(id=compute.findtext('ID'), name=compute.findtext('NAME'), state=state, public_ips=self._extract_networks(compute), private_ips=[], driver=self.connection.driver, image=self._extract_images(compute), size=self._extract_size(compute), extra={'context': self._extract_context(compute)})

    def _extract_networks(self, compute):
        """
        Extract networks from a compute node XML representation.

        Extract network descriptions from a compute node XML representation,
        converting each network to an OpenNebulaNetwork object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``list`` of :class:`OpenNebulaNetwork`
        :return: List of virtual networks attached to the compute node.
        """
        networks = []
        for element in compute.findall('NIC'):
            network = element.find('NETWORK')
            network_id = network.attrib['href'].partition('/network/')[2]
            networks.append(OpenNebulaNetwork(id=network_id, name=network.attrib.get('name', None), address=element.findtext('IP'), size=1, driver=self.connection.driver, extra={'mac': element.findtext('MAC')}))
        return networks

    def _extract_images(self, compute):
        """
        Extract image disks from a compute node XML representation.

        Extract image disk descriptions from a compute node XML representation,
        converting the disks to an NodeImage object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``list`` of :class:`NodeImage`
        :return: Disks attached to a compute node.
        """
        disks = list()
        for element in compute.findall('DISK'):
            disk = element.find('STORAGE')
            image_id = disk.attrib['href'].partition('/storage/')[2]
            if 'id' in element.attrib:
                disk_id = element.attrib['id']
            else:
                disk_id = None
            disks.append(NodeImage(id=image_id, name=disk.attrib.get('name', None), driver=self.connection.driver, extra={'type': element.findtext('TYPE'), 'disk_id': disk_id, 'target': element.findtext('TARGET')}))
        if len(disks) > 1:
            return disks
        elif len(disks) == 1:
            return disks[0]
        else:
            return None

    def _extract_size(self, compute):
        """
        Extract size, or node type, from a compute node XML representation.

        Extract node size, or node type, description from a compute node XML
        representation, converting the node size to a NodeSize object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  :class:`OpenNebulaNodeSize`
        :return: Node type of compute node.
        """
        instance_type = compute.find('INSTANCE_TYPE')
        try:
            return next((node_size for node_size in self.list_sizes() if node_size.name == instance_type.text))
        except StopIteration:
            return None

    def _extract_context(self, compute):
        """
        Extract size, or node type, from a compute node XML representation.

        Extract node size, or node type, description from a compute node XML
        representation, converting the node size to a NodeSize object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``dict``
        :return: Dictionary containing (key, value) pairs related to
                 compute node context.
        """
        contexts = dict()
        context = compute.find('CONTEXT')
        if context is not None:
            for context_element in list(context):
                contexts[context_element.tag.lower()] = context_element.text
        return contexts