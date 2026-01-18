import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_1_0_NodeDriver(OpenStackNodeDriver):
    """
    OpenStack node driver.

    Extra node attributes:
        - password: root password, available after create.
        - hostId: represents the host your cloud server runs on
        - imageId: id of image
        - flavorId: id of flavor
    """
    connectionCls = OpenStack_1_0_Connection
    type = Provider.OPENSTACK
    features = {'create_node': ['generates_password']}

    def __init__(self, *args, **kwargs):
        self._ex_force_api_version = str(kwargs.pop('ex_force_api_version', None))
        self.XML_NAMESPACE = self.connectionCls.XML_NAMESPACE
        super().__init__(*args, **kwargs)

    def _to_images(self, object, ex_only_active):
        images = []
        for image in findall(object, 'image', self.XML_NAMESPACE):
            if ex_only_active and image.get('status') != 'ACTIVE':
                continue
            images.append(self._to_image(image))
        return images

    def _to_image(self, element):
        return NodeImage(id=element.get('id'), name=element.get('name'), driver=self.connection.driver, extra={'updated': element.get('updated'), 'created': element.get('created'), 'status': element.get('status'), 'serverId': element.get('serverId'), 'progress': element.get('progress'), 'minDisk': element.get('minDisk'), 'minRam': element.get('minRam')})

    def _change_password_or_name(self, node, name=None, password=None):
        uri = '/servers/%s' % node.id
        if not name:
            name = node.name
        body = {'xmlns': self.XML_NAMESPACE, 'name': name}
        if password is not None:
            body['adminPass'] = password
        server_elm = ET.Element('server', body)
        resp = self.connection.request(uri, method='PUT', data=ET.tostring(server_elm))
        if resp.status == httplib.NO_CONTENT and password is not None:
            node.extra['password'] = password
        return resp.status == httplib.NO_CONTENT

    def create_node(self, name, size, image, ex_metadata=None, ex_files=None, ex_shared_ip_group=None, ex_shared_ip_group_id=None):
        """
        Create a new node

        @inherits: :class:`NodeDriver.create_node`

        :keyword    ex_metadata: Key/Value metadata to associate with a node
        :type       ex_metadata: ``dict``

        :keyword    ex_files:   File Path => File contents to create on
                                the node
        :type       ex_files:   ``dict``

        :keyword    ex_shared_ip_group_id: The server is launched into
            that shared IP group
        :type       ex_shared_ip_group_id: ``str``
        """
        attributes = {'xmlns': self.XML_NAMESPACE, 'name': name, 'imageId': str(image.id), 'flavorId': str(size.id)}
        if ex_shared_ip_group:
            warnings.warn('ex_shared_ip_group argument is deprecated. Please use ex_shared_ip_group_id')
        if ex_shared_ip_group_id:
            attributes['sharedIpGroupId'] = ex_shared_ip_group_id
        server_elm = ET.Element('server', attributes)
        metadata_elm = self._metadata_to_xml(ex_metadata or {})
        if metadata_elm:
            server_elm.append(metadata_elm)
        files_elm = self._files_to_xml(ex_files or {})
        if files_elm:
            server_elm.append(files_elm)
        resp = self.connection.request('/servers', method='POST', data=ET.tostring(server_elm))
        return self._to_node(resp.object)

    def ex_set_password(self, node, password):
        """
        Sets the Node's root password.

        This will reboot the instance to complete the operation.

        :class:`Node.extra['password']` will be set to the new value if the
        operation was successful.

        :param      node: node to set password
        :type       node: :class:`Node`

        :param      password: new password.
        :type       password: ``str``

        :rtype: ``bool``
        """
        return self._change_password_or_name(node, password=password)

    def ex_set_server_name(self, node, name):
        """
        Sets the Node's name.

        This will reboot the instance to complete the operation.

        :param      node: node to set name
        :type       node: :class:`Node`

        :param      name: new name
        :type       name: ``str``

        :rtype: ``bool``
        """
        return self._change_password_or_name(node, name=name)

    def ex_resize_node(self, node, size):
        """
        Change an existing server flavor / scale the server up or down.

        :param      node: node to resize.
        :type       node: :class:`Node`

        :param      size: new size.
        :type       size: :class:`NodeSize`

        :rtype: ``bool``
        """
        elm = ET.Element('resize', {'xmlns': self.XML_NAMESPACE, 'flavorId': str(size.id)})
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=ET.tostring(elm))
        return resp.status == httplib.ACCEPTED

    def ex_resize(self, node, size):
        """
        NOTE: This method is here for backward compatibility reasons.

        You should use ``ex_resize_node`` instead.
        """
        return self.ex_resize_node(node=node, size=size)

    def ex_confirm_resize(self, node):
        """
        Confirm a resize request which is currently in progress. If a resize
        request is not explicitly confirmed or reverted it's automatically
        confirmed after 24 hours.

        For more info refer to the API documentation: http://goo.gl/zjFI1

        :param      node: node for which the resize request will be confirmed.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        elm = ET.Element('confirmResize', {'xmlns': self.XML_NAMESPACE})
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=ET.tostring(elm))
        return resp.status == httplib.NO_CONTENT

    def ex_revert_resize(self, node):
        """
        Revert a resize request which is currently in progress.
        All resizes are automatically confirmed after 24 hours if they have
        not already been confirmed explicitly or reverted.

        For more info refer to the API documentation: http://goo.gl/AizBu

        :param      node: node for which the resize request will be reverted.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        elm = ET.Element('revertResize', {'xmlns': self.XML_NAMESPACE})
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=ET.tostring(elm))
        return resp.status == httplib.NO_CONTENT

    def ex_rebuild(self, node_id, image_id):
        """
        Rebuilds the specified server.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :param       image_id: ID of the image which should be used
        :type        image_id: ``str``

        :rtype: ``bool``
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        if isinstance(image_id, NodeImage):
            image_id = image_id.id
        elm = ET.Element('rebuild', {'xmlns': self.XML_NAMESPACE, 'imageId': image_id})
        resp = self.connection.request('/servers/%s/action' % node_id, method='POST', data=ET.tostring(elm))
        return resp.status == httplib.ACCEPTED

    def ex_create_ip_group(self, group_name, node_id=None):
        """
        Creates a shared IP group.

        :param       group_name:  group name which should be used
        :type        group_name: ``str``

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: ``bool``
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        group_elm = ET.Element('sharedIpGroup', {'xmlns': self.XML_NAMESPACE, 'name': group_name})
        if node_id:
            ET.SubElement(group_elm, 'server', {'id': node_id})
        resp = self.connection.request('/shared_ip_groups', method='POST', data=ET.tostring(group_elm))
        return self._to_shared_ip_group(resp.object)

    def ex_list_ip_groups(self, details=False):
        """
        Lists IDs and names for shared IP groups.
        If details lists all details for shared IP groups.

        :param       details: True if details is required
        :type        details: ``bool``

        :rtype: ``list`` of :class:`OpenStack_1_0_SharedIpGroup`
        """
        uri = '/shared_ip_groups/detail' if details else '/shared_ip_groups'
        resp = self.connection.request(uri, method='GET')
        groups = findall(resp.object, 'sharedIpGroup', self.XML_NAMESPACE)
        return [self._to_shared_ip_group(el) for el in groups]

    def ex_delete_ip_group(self, group_id):
        """
        Deletes the specified shared IP group.

        :param       group_id:  group id which should be used
        :type        group_id: ``str``

        :rtype: ``bool``
        """
        uri = '/shared_ip_groups/%s' % group_id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def ex_share_ip(self, group_id, node_id, ip, configure_node=True):
        """
        Shares an IP address to the specified server.

        :param       group_id:  group id which should be used
        :type        group_id: ``str``

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :param       ip: ip which should be used
        :type        ip: ``str``

        :param       configure_node: configure node
        :type        configure_node: ``bool``

        :rtype: ``bool``
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        if configure_node:
            str_configure = 'true'
        else:
            str_configure = 'false'
        elm = ET.Element('shareIp', {'xmlns': self.XML_NAMESPACE, 'sharedIpGroupId': group_id, 'configureServer': str_configure})
        uri = '/servers/{}/ips/public/{}'.format(node_id, ip)
        resp = self.connection.request(uri, method='PUT', data=ET.tostring(elm))
        return resp.status == httplib.ACCEPTED

    def ex_unshare_ip(self, node_id, ip):
        """
        Removes a shared IP address from the specified server.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :param       ip: ip which should be used
        :type        ip: ``str``

        :rtype: ``bool``
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        uri = '/servers/{}/ips/public/{}'.format(node_id, ip)
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.ACCEPTED

    def ex_list_ip_addresses(self, node_id):
        """
        List all server addresses.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: :class:`OpenStack_1_0_NodeIpAddresses`
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        uri = '/servers/%s/ips' % node_id
        resp = self.connection.request(uri, method='GET')
        return self._to_ip_addresses(resp.object)

    def _metadata_to_xml(self, metadata):
        if not metadata:
            return None
        metadata_elm = ET.Element('metadata')
        for k, v in list(metadata.items()):
            meta_elm = ET.SubElement(metadata_elm, 'meta', {'key': str(k)})
            meta_elm.text = str(v)
        return metadata_elm

    def _files_to_xml(self, files):
        if not files:
            return None
        personality_elm = ET.Element('personality')
        for k, v in list(files.items()):
            file_elm = ET.SubElement(personality_elm, 'file', {'path': str(k)})
            file_elm.text = base64.b64encode(b(v)).decode('ascii')
        return personality_elm

    def _reboot_node(self, node, reboot_type='SOFT'):
        resp = self._node_action(node, ['reboot', ('type', reboot_type)])
        return resp.status == httplib.ACCEPTED

    def _node_action(self, node, body):
        if isinstance(body, list):
            attr = ' '.join(['{}="{}"'.format(item[0], item[1]) for item in body[1:]])
            body = '<{} xmlns="{}" {}/>'.format(body[0], self.XML_NAMESPACE, attr)
        uri = '/servers/%s/action' % node.id
        resp = self.connection.request(uri, method='POST', data=body)
        return resp

    def _to_nodes(self, object):
        node_elements = findall(object, 'server', self.XML_NAMESPACE)
        return [self._to_node(el) for el in node_elements]

    def _to_node_from_obj(self, obj):
        return self._to_node(findall(obj, 'server', self.XML_NAMESPACE)[0])

    def _to_node(self, el):

        def get_ips(el):
            return [ip.get('addr') for ip in el]

        def get_meta_dict(el):
            d = {}
            for meta in el:
                d[meta.get('key')] = meta.text
            return d
        public_ip = get_ips(findall(el, 'addresses/public/ip', self.XML_NAMESPACE))
        private_ip = get_ips(findall(el, 'addresses/private/ip', self.XML_NAMESPACE))
        metadata = get_meta_dict(findall(el, 'metadata/meta', self.XML_NAMESPACE))
        n = Node(id=el.get('id'), name=el.get('name'), state=self.NODE_STATE_MAP.get(el.get('status'), NodeState.UNKNOWN), public_ips=public_ip, private_ips=private_ip, driver=self.connection.driver, extra={'password': el.get('adminPass'), 'hostId': el.get('hostId'), 'imageId': el.get('imageId'), 'flavorId': el.get('flavorId'), 'uri': 'https://%s%s/servers/%s' % (self.connection.host, self.connection.request_path, el.get('id')), 'service_name': self.connection.get_service_name(), 'metadata': metadata})
        return n

    def _to_sizes(self, object):
        elements = findall(object, 'flavor', self.XML_NAMESPACE)
        return [self._to_size(el) for el in elements]

    def _to_size(self, el):
        vcpus = int(el.get('vcpus')) if el.get('vcpus', None) else None
        return OpenStackNodeSize(id=el.get('id'), name=el.get('name'), ram=int(el.get('ram')), disk=int(el.get('disk')), vcpus=vcpus, bandwidth=None, extra=el.get('extra_specs'), price=self._get_size_price(el.get('id')), driver=self.connection.driver)

    def ex_limits(self):
        """
        Extra call to get account's limits, such as
        rates (for example amount of POST requests per day)
        and absolute limits like total amount of available
        RAM to be used by servers.

        :return: dict with keys 'rate' and 'absolute'
        :rtype: ``dict``
        """

        def _to_rate(el):
            rate = {}
            for item in list(el.items()):
                rate[item[0]] = item[1]
            return rate

        def _to_absolute(el):
            return {el.get('name'): el.get('value')}
        limits = self.connection.request('/limits').object
        rate = [_to_rate(el) for el in findall(limits, 'rate/limit', self.XML_NAMESPACE)]
        absolute = {}
        for item in findall(limits, 'absolute/limit', self.XML_NAMESPACE):
            absolute.update(_to_absolute(item))
        return {'rate': rate, 'absolute': absolute}

    def create_image(self, node, name, description=None, reboot=True):
        """Create an image for node.

        @inherits: :class:`NodeDriver.create_image`

        :param      node: node to use as a base for image
        :type       node: :class:`Node`

        :param      name: name for new image
        :type       name: ``str``

        :rtype: :class:`NodeImage`
        """
        image_elm = ET.Element('image', {'xmlns': self.XML_NAMESPACE, 'name': name, 'serverId': node.id})
        return self._to_image(self.connection.request('/images', method='POST', data=ET.tostring(image_elm)).object)

    def delete_image(self, image):
        """Delete an image for node.

        @inherits: :class:`NodeDriver.delete_image`

        :param      image: the image to be deleted
        :type       image: :class:`NodeImage`

        :rtype: ``bool``
        """
        uri = '/images/%s' % image.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def _to_shared_ip_group(self, el):
        servers_el = findall(el, 'servers', self.XML_NAMESPACE)
        if servers_el:
            servers = [s.get('id') for s in findall(servers_el[0], 'server', self.XML_NAMESPACE)]
        else:
            servers = None
        return OpenStack_1_0_SharedIpGroup(id=el.get('id'), name=el.get('name'), servers=servers)

    def _to_ip_addresses(self, el):
        public_ips = [ip.get('addr') for ip in findall(findall(el, 'public', self.XML_NAMESPACE)[0], 'ip', self.XML_NAMESPACE)]
        private_ips = [ip.get('addr') for ip in findall(findall(el, 'private', self.XML_NAMESPACE)[0], 'ip', self.XML_NAMESPACE)]
        return OpenStack_1_0_NodeIpAddresses(public_ips, private_ips)

    def _get_size_price(self, size_id):
        try:
            return get_size_price(driver_type='compute', driver_name=self.api_name, size_id=size_id)
        except KeyError:
            return 0.0