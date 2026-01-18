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
class OpenStack_2_NodeDriver(OpenStack_1_1_NodeDriver):
    """
    OpenStack node driver.
    """
    connectionCls = OpenStack_2_Connection
    image_connectionCls = OpenStack_2_ImageConnection
    image_connection = None
    network_connectionCls = OpenStack_2_NetworkConnection
    network_connection = None
    volumev2_connectionCls = OpenStack_2_VolumeV2Connection
    volumev3_connectionCls = OpenStack_2_VolumeV3Connection
    volumev2_connection = None
    volumev3_connection = None
    volume_connection = None
    type = Provider.OPENSTACK
    features = {'create_node': ['generates_password']}
    _networks_url_prefix = '/v2.0/networks'
    _subnets_url_prefix = '/v2.0/subnets'
    PORT_INTERFACE_MAP = {'BUILD': OpenStack_2_PortInterfaceState.BUILD, 'ACTIVE': OpenStack_2_PortInterfaceState.ACTIVE, 'DOWN': OpenStack_2_PortInterfaceState.DOWN, 'UNKNOWN': OpenStack_2_PortInterfaceState.UNKNOWN}

    def __init__(self, *args, **kwargs):
        original_connectionCls = self.connectionCls
        self._ex_force_api_version = str(kwargs.pop('ex_force_api_version', None))
        if 'ex_force_auth_version' not in kwargs:
            kwargs['ex_force_auth_version'] = '3.x_password'
        original_ex_force_base_url = kwargs.get('ex_force_base_url')
        if original_ex_force_base_url or kwargs.get('ex_force_image_url'):
            kwargs['ex_force_base_url'] = str(kwargs.pop('ex_force_image_url', original_ex_force_base_url))
        self.connectionCls = self.image_connectionCls
        super().__init__(*args, **kwargs)
        self.image_connection = self.connection
        if original_ex_force_base_url or kwargs.get('ex_force_volume_url'):
            kwargs['ex_force_base_url'] = str(kwargs.pop('ex_force_volume_url', original_ex_force_base_url))
        self.connectionCls = self.volumev3_connectionCls
        super().__init__(*args, **kwargs)
        self.volumev3_connection = self.connection
        self.connectionCls = self.volumev2_connectionCls
        super().__init__(*args, **kwargs)
        self.volumev2_connection = self.connection
        if original_ex_force_base_url or kwargs.get('ex_force_network_url'):
            kwargs['ex_force_base_url'] = str(kwargs.pop('ex_force_network_url', original_ex_force_base_url))
        self.connectionCls = self.network_connectionCls
        super().__init__(*args, **kwargs)
        self.network_connection = self.connection
        self._ex_force_base_url = original_ex_force_base_url
        if original_ex_force_base_url:
            kwargs['ex_force_base_url'] = self._ex_force_base_url
        elif 'ex_force_base_url' in kwargs:
            del kwargs['ex_force_base_url']
        self.connectionCls = original_connectionCls
        super().__init__(*args, **kwargs)

    def _to_port(self, element):
        created = element.get('created_at')
        updated = element.get('updated_at')
        return OpenStack_2_PortInterface(id=element['id'], state=self.PORT_INTERFACE_MAP.get(element.get('status'), OpenStack_2_PortInterfaceState.UNKNOWN), created=created, driver=self, extra=dict(admin_state_up=element.get('admin_state_up'), allowed_address_pairs=element.get('allowed_address_pairs'), binding_vnic_type=element.get('binding:vnic_type'), binding_host_id=element.get('binding:host_id', None), device_id=element.get('device_id'), description=element.get('description', None), device_owner=element.get('device_owner'), fixed_ips=element.get('fixed_ips'), mac_address=element.get('mac_address'), name=element.get('name'), network_id=element.get('network_id'), project_id=element.get('project_id', None), port_security_enabled=element.get('port_security_enabled', None), revision_number=element.get('revision_number', None), security_groups=element.get('security_groups'), tags=element.get('tags', None), tenant_id=element.get('tenant_id'), updated=updated))

    def list_nodes(self, ex_all_tenants=False):
        """
        List the nodes in a tenant

        :param ex_all_tenants: List nodes for all the tenants. Note: Your user
                               must have admin privileges for this
                               functionality to work.
        :type ex_all_tenants: ``bool``
        """
        params = {}
        if ex_all_tenants:
            params = {'all_tenants': 1}
        return self._to_nodes(self._paginated_request('/servers/detail', 'servers', self.connection, params=params))

    def get_image(self, image_id):
        """
        Get a NodeImage using the V2 Glance API

        @inherits: :class:`OpenStack_1_1_NodeDriver.get_image`

        :param      image_id: ID of the image which should be used
        :type       image_id: ``str``

        :rtype: :class:`NodeImage`
        """
        return self._to_image(self.image_connection.request('/v2/images/{}'.format(image_id)).object)

    def list_images(self, location=None, ex_only_active=True):
        """
        Lists all active images using the V2 Glance API

        @inherits: :class:`NodeDriver.list_images`

        :param location: Which data center to list the images in. If
                               empty, undefined behavior will be selected.
                               (optional)
        :type location: :class:`.NodeLocation`

        :param ex_only_active: True if list only active (optional)
        :type ex_only_active: ``bool``
        """
        if location is not None:
            raise NotImplementedError('location in list_images is not implemented in the OpenStack_2_NodeDriver')
        if not ex_only_active:
            raise NotImplementedError('ex_only_active in list_images is not implemented in the OpenStack_2_NodeDriver')
        result = self._paginated_request_next(path='/v2/images', request_method=self.image_connection.request, response_key='images')
        images = []
        for item in result:
            images.append(self._to_image(item))
        return images

    def ex_update_image(self, image_id, data):
        """
        Patch a NodeImage. Can be used to set visibility

        :param      image_id: ID of the image which should be used
        :type       image_id: ``str``

        :param      data: The data to PATCH, either a dict or a list
        for example: [
          {'op': 'replace', 'path': '/visibility', 'value': 'shared'}
        ]
        :type       data: ``dict|list``

        :rtype: :class:`NodeImage`
        """
        response = self.image_connection.request('/v2/images/{}'.format(image_id), headers={'Content-type': 'application/openstack-images-v2.1-json-patch'}, method='PATCH', data=data)
        return self._to_image(response.object)

    def ex_list_image_members(self, image_id):
        """
        List all members of an image. See
        https://developer.openstack.org/api-ref/image/v2/index.html#sharing

        :param      image_id: ID of the image of which the members should
        be listed
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
        response = self.image_connection.request('/v2/images/{}/members'.format(image_id))
        image_members = []
        for image_member in response.object['members']:
            image_members.append(self._to_image_member(image_member))
        return image_members

    def ex_create_image_member(self, image_id, member_id):
        """
        Give a project access to an image.

        The image should have visibility status 'shared'.

        Note that this is not an idempotent operation. If this action is
        attempted using a tenant that is already in the image members
        group the API will throw a Conflict (409).
        See the 'create-image-member' section on
        https://developer.openstack.org/api-ref/image/v2/index.html

        :param str image_id: The ID of the image to share with the specified
        tenant
        :param str member_id: The ID of the project / tenant (the image member)
        Note that this is the Keystone project ID and not the project name,
        so something like e2151b1fe02d4a8a2d1f5fc331522c0a
        :return None:

        :param      image_id: ID of the image to share
        :type       image_id: ``str``

        :param      project: ID of the project to give access to the image
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
        data = {'member': member_id}
        response = self.image_connection.request('/v2/images/%s/members' % image_id, method='POST', data=data)
        return self._to_image_member(response.object)

    def ex_get_image_member(self, image_id, member_id):
        """
        Get a member of an image by id

        :param      image_id: ID of the image of which the member should
        be listed
        :type       image_id: ``str``

        :param      member_id: ID of the member to list
        :type       image_id: ``str``

        :rtype: ``list`` of :class:`NodeImageMember`
        """
        response = self.image_connection.request('/v2/images/{}/members/{}'.format(image_id, member_id))
        return self._to_image_member(response.object)

    def ex_accept_image_member(self, image_id, member_id):
        """
        Accept a pending image as a member.

        This call is idempotent unlike ex_create_image_member,
        you can accept the same image many times.

        :param      image_id: ID of the image to accept
        :type       image_id: ``str``

        :param      project: ID of the project to accept the image as
        :type       image_id: ``str``

        :rtype: ``bool``
        """
        data = {'status': 'accepted'}
        response = self.image_connection.request('/v2/images/{}/members/{}'.format(image_id, member_id), method='PUT', data=data)
        return self._to_image_member(response.object)

    def _to_networks(self, obj):
        networks = obj['networks']
        return [self._to_network(network) for network in networks]

    def _to_network(self, obj):
        extra = {}
        if obj.get('router:external', None):
            extra['router:external'] = obj.get('router:external')
        if obj.get('subnets', None):
            extra['subnets'] = obj.get('subnets')
        return OpenStackNetwork(id=obj['id'], name=obj['name'], cidr=None, driver=self, extra=extra)

    def ex_list_networks(self):
        """
        Get a list of Networks that are available.

        :rtype: ``list`` of :class:`OpenStackNetwork`
        """
        response = self.network_connection.request(self._networks_url_prefix).object
        return self._to_networks(response)

    def ex_get_network(self, network_id):
        """
        Retrieve the Network with the given ID

        :param networkId: ID of the network
        :type networkId: ``str``

        :rtype :class:`OpenStackNetwork`
        """
        request_url = '{networks_url_prefix}/{network_id}'.format(networks_url_prefix=self._networks_url_prefix, network_id=network_id)
        response = self.network_connection.request(request_url).object
        return self._to_network(response['network'])

    def ex_create_network(self, name, **kwargs):
        """
        Create a new Network

        :param name: Name of network which should be used
        :type name: ``str``

        :rtype: :class:`OpenStackNetwork`
        """
        data = {'network': {'name': name}}
        for key, value in kwargs.items():
            data['network'][key] = value
        response = self.network_connection.request(self._networks_url_prefix, method='POST', data=data).object
        return self._to_network(response['network'])

    def ex_delete_network(self, network):
        """
        Delete a Network

        :param network: Network which should be used
        :type network: :class:`OpenStackNetwork`

        :rtype: ``bool``
        """
        resp = self.network_connection.request('{}/{}'.format(self._networks_url_prefix, network.id), method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def _to_subnets(self, obj):
        subnets = obj['subnets']
        return [self._to_subnet(subnet) for subnet in subnets]

    def _to_subnet(self, obj):
        extra = {}
        if obj.get('router:external', None):
            extra['router:external'] = obj.get('router:external')
        if obj.get('subnets', None):
            extra['subnets'] = obj.get('subnets')
        return OpenStack_2_SubNet(id=obj['id'], name=obj['name'], cidr=obj['cidr'], network_id=obj['network_id'], driver=self, extra=extra)

    def ex_list_subnets(self):
        """
        Get a list of Subnet that are available.

        :rtype: ``list`` of :class:`OpenStack_2_SubNet`
        """
        response = self.network_connection.request(self._subnets_url_prefix).object
        return self._to_subnets(response)

    def ex_create_subnet(self, name, network, cidr, ip_version=4, description='', dns_nameservers=None, host_routes=None):
        """
        Create a new Subnet

        :param name: Name of subnet which should be used
        :type name: ``str``

        :param network: Parent network of the subnet
        :type network: ``OpenStackNetwork``

        :param cidr: cidr of network which should be used
        :type cidr: ``str``

        :param ip_version: ip_version of subnet which should be used
        :type ip_version: ``int``

        :param description: Description for the resource.
        :type description: ``str``

        :param dns_nameservers: List of dns name servers.
        :type dns_nameservers: ``list`` of ``str``

        :param host_routes: Additional routes for the subnet.
        :type host_routes: ``list`` of ``str``

        :rtype: :class:`OpenStack_2_SubNet`
        """
        data = {'subnet': {'cidr': cidr, 'network_id': network.id, 'ip_version': ip_version, 'name': name or '', 'description': description or '', 'dns_nameservers': dns_nameservers or [], 'host_routes': host_routes or []}}
        response = self.network_connection.request(self._subnets_url_prefix, method='POST', data=data).object
        return self._to_subnet(response['subnet'])

    def ex_delete_subnet(self, subnet):
        """
        Delete a Subnet

        :param subnet: Subnet which should be deleted
        :type subnet: :class:`OpenStack_2_SubNet`

        :rtype: ``bool``
        """
        resp = self.network_connection.request('{}/{}'.format(self._subnets_url_prefix, subnet.id), method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def ex_update_subnet(self, subnet, name=None, description=None, dns_nameservers=None, host_routes=None):
        """
        Update data of an existing SubNet

        :param subnet: Subnet which should be updated
        :type subnet: :class:`OpenStack_2_SubNet`

        :param name: Name of subnet which should be used
        :type name: ``str``

        :param description: Description for the resource.
        :type description: ``str``

        :param dns_nameservers: List of dns name servers.
        :type dns_nameservers: ``list`` of ``str``

        :param host_routes: Additional routes for the subnet.
        :type host_routes: ``list`` of ``str``

        :rtype: :class:`OpenStack_2_SubNet`
        """
        data = {'subnet': {}}
        if name is not None:
            data['subnet']['name'] = name
        if description is not None:
            data['subnet']['description'] = description
        if dns_nameservers is not None:
            data['subnet']['dns_nameservers'] = dns_nameservers
        if host_routes is not None:
            data['subnet']['host_routes'] = host_routes
        response = self.network_connection.request('{}/{}'.format(self._subnets_url_prefix, subnet.id), method='PUT', data=data).object
        return self._to_subnet(response['subnet'])

    def ex_list_ports(self):
        """
        List all OpenStack_2_PortInterfaces

        https://developer.openstack.org/api-ref/network/v2/#list-ports

        :rtype: ``list`` of :class:`OpenStack_2_PortInterface`
        """
        response = self._paginated_request('/v2.0/ports', 'ports', self.network_connection)
        return [self._to_port(port) for port in response['ports']]

    def ex_delete_port(self, port):
        """
        Delete an OpenStack_2_PortInterface

        https://developer.openstack.org/api-ref/network/v2/#delete-port

        :param      port: port interface to remove
        :type       port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        response = self.network_connection.request('/v2.0/ports/%s' % port.id, method='DELETE')
        return response.success()

    def ex_detach_port_interface(self, node, port):
        """
        Detaches an OpenStack_2_PortInterface interface from a Node.
        :param      node: node
        :type       node: :class:`Node`

        :param      port: port interface to detach
        :type       port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        return self.connection.request('/servers/{}/os-interface/{}'.format(node.id, port.id), method='DELETE').success()

    def ex_attach_port_interface(self, node, port):
        """
        Attaches an OpenStack_2_PortInterface to a Node.

        :param      node: node
        :type       node: :class:`Node`

        :param      port: port interface to attach
        :type       port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        data = {'interfaceAttachment': {'port_id': port.id}}
        return self.connection.request('/servers/{}/os-interface'.format(node.id), method='POST', data=data).success()

    def ex_create_port(self, network, description=None, admin_state_up=True, name=None):
        """
        Creates a new OpenStack_2_PortInterface

        :param      network: ID of the network where the newly created
                    port should be attached to
        :type       network: :class:`OpenStackNetwork`

        :param      description: Description of the port
        :type       description: str

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: bool

        :param      name: Human-readable name of the resource
        :type       name: str

        :rtype: :class:`OpenStack_2_PortInterface`
        """
        data = {'port': {'description': description or '', 'admin_state_up': admin_state_up, 'name': name or '', 'network_id': network.id}}
        response = self.network_connection.request('/v2.0/ports', method='POST', data=data)
        return self._to_port(response.object['port'])

    def ex_get_port(self, port_interface_id):
        """
        Retrieve the OpenStack_2_PortInterface with the given ID

        :param      port_interface_id: ID of the requested port
        :type       port_interface_id: str

        :return: :class:`OpenStack_2_PortInterface`
        """
        response = self.network_connection.request('/v2.0/ports/{}'.format(port_interface_id), method='GET')
        return self._to_port(response.object['port'])

    def ex_update_port(self, port, description=None, admin_state_up=None, name=None, port_security_enabled=None, qos_policy_id=None, security_groups=None, allowed_address_pairs=None):
        """
        Update a OpenStack_2_PortInterface

        :param      port: port interface to update
        :type       port: :class:`OpenStack_2_PortInterface`

        :param      description: Description of the port
        :type       description: ``str``

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: ``bool``

        :param      name: Human-readable name of the resource
        :type       name: ``str``

        :param      port_security_enabled: 	The port security status
        :type       port_security_enabled: ``bool``

        :param      qos_policy_id: QoS policy associated with the port
        :type       qos_policy_id: ``str``

        :param      security_groups: The IDs of security groups applied
        :type       security_groups: ``list`` of ``str``

        :param      allowed_address_pairs: IP and MAC address that the port
                    can use when sending packets if port_security_enabled is
                    true
        :type       allowed_address_pairs: ``list`` of ``dict`` containing
                    ip_address and mac_address; mac_address is optional, taken
                    from the port if not specified

        :rtype: :class:`OpenStack_2_PortInterface`
        """
        data = {'port': {}}
        if description is not None:
            data['port']['description'] = description
        if admin_state_up is not None:
            data['port']['admin_state_up'] = admin_state_up
        if name is not None:
            data['port']['name'] = name
        if port_security_enabled is not None:
            data['port']['port_security_enabled'] = port_security_enabled
        if qos_policy_id is not None:
            data['port']['qos_policy_id'] = qos_policy_id
        if security_groups is not None:
            data['port']['security_groups'] = security_groups
        if allowed_address_pairs is not None:
            data['port']['allowed_address_pairs'] = allowed_address_pairs
        response = self.network_connection.request('/v2.0/ports/{}'.format(port.id), method='PUT', data=data)
        return self._to_port(response.object['port'])

    def ex_get_node_ports(self, node):
        """
        Get the list of OpenStack_2_PortInterface interfaces from a Node.
        :param      node: node
        :type       node: :class:`Node`

        :rtype: ``list`` of :class:`OpenStack_2_PortInterface`
        """
        response = self.connection.request('/servers/%s/os-interface' % node.id, method='GET')
        ports = []
        for port in response.object['interfaceAttachments']:
            port['id'] = port.pop('port_id')
            ports.append(self._to_port(port))
        return ports

    def _get_volume_connection(self):
        """
        Get the correct Volume connection (v3 or v2)
        """
        if not self.volume_connection:
            try:
                self.volumev3_connection.get_service_catalog()
                self.volume_connection = self.volumev3_connection
            except LibcloudError:
                self.volume_connection = self.volumev2_connection
        return self.volume_connection

    def list_volumes(self):
        """
        Get a list of Volumes that are available.

        :rtype: ``list`` of :class:`StorageVolume`
        """
        return self._to_volumes(self._paginated_request('/volumes/detail', 'volumes', self._get_volume_connection()))

    def ex_get_volume(self, volumeId):
        """
        Retrieve the StorageVolume with the given ID

        :param volumeId: ID of the volume
        :type volumeId: ``string``

        :return: :class:`StorageVolume`
        """
        return self._to_volume(self._get_volume_connection().request('/volumes/%s' % volumeId).object)

    def create_volume(self, size, name, location=None, snapshot=None, ex_volume_type=None, ex_image_ref=None):
        """
        Create a new volume.

        :param size: Size of volume in gigabytes (required)
        :type size: ``int``

        :param name: Name of the volume to be created
        :type name: ``str``

        :param location: Which data center to create a volume in. If
                               empty, undefined behavior will be selected.
                               (optional)
        :type location: :class:`.NodeLocation`

        :param snapshot:  Snapshot from which to create the new
                          volume.  (optional)
        :type snapshot:  :class:`.VolumeSnapshot`

        :param ex_volume_type: What kind of volume to create.
                            (optional)
        :type ex_volume_type: ``str``

        :param ex_image_ref: The image to create the volume from
                             when creating a bootable volume (optional)
        :type ex_image_ref: ``str``

        :return: The newly created volume.
        :rtype: :class:`StorageVolume`
        """
        volume = {'name': name, 'description': name, 'size': size, 'metadata': {'contents': name}}
        if ex_volume_type:
            volume['volume_type'] = ex_volume_type
        if ex_image_ref:
            volume['imageRef'] = ex_image_ref
        if location:
            volume['availability_zone'] = location
        if snapshot:
            volume['snapshot_id'] = snapshot.id
        resp = self._get_volume_connection().request('/volumes', method='POST', data={'volume': volume})
        return self._to_volume(resp.object)

    def destroy_volume(self, volume):
        """
        Delete a Volume.

        :param volume: Volume to be deleted
        :type  volume: :class:`StorageVolume`

        :rtype: ``bool``
        """
        return self._get_volume_connection().request('/volumes/%s' % volume.id, method='DELETE').success()

    def ex_list_snapshots(self):
        """
        Get a list of Snapshot that are available.

        :rtype: ``list`` of :class:`VolumeSnapshot`
        """
        return self._to_snapshots(self._paginated_request('/snapshots/detail', 'snapshots', self._get_volume_connection()))

    def create_volume_snapshot(self, volume, name=None, ex_description=None, ex_force=True):
        """
        Create snapshot from volume

        :param volume: Instance of `StorageVolume`
        :type  volume: `StorageVolume`

        :param name: Name of snapshot (optional)
        :type  name: `str` | `NoneType`

        :param ex_description: Description of the snapshot (optional)
        :type  ex_description: `str` | `NoneType`

        :param ex_force: Specifies if we create a snapshot that is not in
                         state `available`. For example `in-use`. Defaults
                         to True. (optional)
        :type  ex_force: `bool`

        :rtype: :class:`VolumeSnapshot`
        """
        data = {'snapshot': {'volume_id': volume.id, 'force': ex_force}}
        if name is not None:
            data['snapshot']['name'] = name
        if ex_description is not None:
            data['snapshot']['description'] = ex_description
        return self._to_snapshot(self._get_volume_connection().request('/snapshots', method='POST', data=data).object)

    def destroy_volume_snapshot(self, snapshot):
        """
        Delete a Volume Snapshot.

        :param snapshot: Snapshot to be deleted
        :type  snapshot: :class:`VolumeSnapshot`

        :rtype: ``bool``
        """
        resp = self._get_volume_connection().request('/snapshots/%s' % snapshot.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def ex_list_security_groups(self):
        """
        Get a list of Security Groups that are available.

        :rtype: ``list`` of :class:`OpenStackSecurityGroup`
        """
        return self._to_security_groups(self.network_connection.request('/v2.0/security-groups').object)

    def ex_create_security_group(self, name, description):
        """
        Create a new Security Group

        :param name: Name of the new Security Group
        :type  name: ``str``

        :param description: Description of the new Security Group
        :type  description: ``str``

        :rtype: :class:`OpenStackSecurityGroup`
        """
        return self._to_security_group(self.network_connection.request('/v2.0/security-groups', method='POST', data={'security_group': {'name': name, 'description': description}}).object['security_group'])

    def ex_delete_security_group(self, security_group):
        """
        Delete a Security Group.

        :param security_group: Security Group should be deleted
        :type  security_group: :class:`OpenStackSecurityGroup`

        :rtype: ``bool``
        """
        resp = self.network_connection.request('/v2.0/security-groups/%s' % security_group.id, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def _to_security_group_rule(self, obj):
        ip_range = group = tenant_id = parent_id = None
        protocol = from_port = to_port = direction = None
        if 'parent_group_id' in obj:
            if obj['group'] == {}:
                ip_range = obj['ip_range'].get('cidr', None)
            else:
                group = obj['group'].get('name', None)
                tenant_id = obj['group'].get('tenant_id', None)
            parent_id = obj['parent_group_id']
            from_port = obj['from_port']
            to_port = obj['to_port']
            protocol = obj['ip_protocol']
        else:
            ip_range = obj.get('remote_ip_prefix', None)
            group = obj.get('remote_group_id', None)
            tenant_id = obj.get('tenant_id', None)
            parent_id = obj['security_group_id']
            from_port = obj['port_range_min']
            to_port = obj['port_range_max']
            protocol = obj['protocol']
        return OpenStackSecurityGroupRule(id=obj['id'], parent_group_id=parent_id, ip_protocol=protocol, from_port=from_port, to_port=to_port, driver=self, ip_range=ip_range, group=group, tenant_id=tenant_id, direction=direction)

    def ex_create_security_group_rule(self, security_group, ip_protocol, from_port, to_port, cidr=None, source_security_group=None):
        """
        Create a new Rule in a Security Group

        :param security_group: Security Group in which to add the rule
        :type  security_group: :class:`OpenStackSecurityGroup`

        :param ip_protocol: Protocol to which this rule applies
                            Examples: tcp, udp, ...
        :type  ip_protocol: ``str``

        :param from_port: First port of the port range
        :type  from_port: ``int``

        :param to_port: Last port of the port range
        :type  to_port: ``int``

        :param cidr: CIDR notation of the source IP range for this rule
        :type  cidr: ``str``

        :param source_security_group: Existing Security Group to use as the
                                      source (instead of CIDR)
        :type  source_security_group: L{OpenStackSecurityGroup

        :rtype: :class:`OpenStackSecurityGroupRule`
        """
        source_security_group_id = None
        if type(source_security_group) == OpenStackSecurityGroup:
            source_security_group_id = source_security_group.id
        return self._to_security_group_rule(self.network_connection.request('/v2.0/security-group-rules', method='POST', data={'security_group_rule': {'direction': 'ingress', 'protocol': ip_protocol, 'port_range_min': from_port, 'port_range_max': to_port, 'remote_ip_prefix': cidr, 'remote_group_id': source_security_group_id, 'security_group_id': security_group.id}}).object['security_group_rule'])

    def ex_delete_security_group_rule(self, rule):
        """
        Delete a Rule from a Security Group.

        :param rule: Rule should be deleted
        :type  rule: :class:`OpenStackSecurityGroupRule`

        :rtype: ``bool``
        """
        resp = self.network_connection.request('/v2.0/security-group-rules/%s' % rule.id, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def ex_remove_security_group_from_node(self, security_group, node):
        """
        Remove a Security Group from a node.

        :param security_group: Security Group to remove from node.
        :type  security_group: :class:`OpenStackSecurityGroup`

        :param      node: Node to remove the Security Group.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        server_params = {'name': security_group.name}
        resp = self._node_action(node, 'removeSecurityGroup', **server_params)
        return resp.status == httplib.ACCEPTED

    def _to_floating_ip_pool(self, obj):
        return OpenStack_2_FloatingIpPool(obj['id'], obj['name'], self.network_connection)

    def _to_floating_ip_pools(self, obj):
        pool_elements = obj['networks']
        return [self._to_floating_ip_pool(pool) for pool in pool_elements]

    def ex_list_floating_ip_pools(self):
        """
        List available floating IP pools

        :rtype: ``list`` of :class:`OpenStack_2_FloatingIpPool`
        """
        return self._to_floating_ip_pools(self.network_connection.request('/v2.0/networks?router:external=True&fields=id&fields=name').object)

    def _to_routers(self, obj):
        routers = obj['routers']
        return [self._to_router(router) for router in routers]

    def _to_router(self, obj):
        extra = {}
        extra['external_gateway_info'] = obj['external_gateway_info']
        extra['routes'] = obj['routes']
        return OpenStack_2_Router(id=obj['id'], name=obj['name'], status=obj['status'], driver=self, extra=extra)

    def ex_list_routers(self):
        """
        Get a list of Routers that are available.

        :rtype: ``list`` of :class:`OpenStack_2_Router`
        """
        response = self.network_connection.request('/v2.0/routers').object
        return self._to_routers(response)

    def ex_create_router(self, name, description='', admin_state_up=True, external_gateway_info=None):
        """
        Create a new Router

        :param name: Name of router which should be used
        :type name: ``str``

        :param      description: Description of the port
        :type       description: ``str``

        :param      admin_state_up: The administrative state of the
                    resource, which is up or down
        :type       admin_state_up: ``bool``

        :param      external_gateway_info: The external gateway information
        :type       external_gateway_info: ``dict``

        :rtype: :class:`OpenStack_2_Router`
        """
        data = {'router': {'name': name or '', 'description': description or '', 'admin_state_up': admin_state_up}}
        if external_gateway_info:
            data['router']['external_gateway_info'] = external_gateway_info
        response = self.network_connection.request('/v2.0/routers', method='POST', data=data).object
        return self._to_router(response['router'])

    def ex_delete_router(self, router):
        """
        Delete a Router

        :param router: Router which should be deleted
        :type router: :class:`OpenStack_2_Router`

        :rtype: ``bool``
        """
        resp = self.network_connection.request('{}/{}'.format('/v2.0/routers', router.id), method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def _manage_router_interface(self, router, op, subnet=None, port=None):
        """
        Add/Remove interface to router

        :param router: Router to add/remove the interface
        :type router: :class:`OpenStack_2_Router`

        :param      op: Operation to perform: 'add' or 'remove'
        :type       op: ``str``

        :param subnet: Subnet object to be added to the router
        :type subnet: :class:`OpenStack_2_SubNet`

        :param port: Port object to be added to the router
        :type port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        data = {}
        if subnet:
            data['subnet_id'] = subnet.id
        elif port:
            data['port_id'] = port.id
        else:
            raise OpenStackException('Error in router interface: port or subnet are None.', 500, self)
        resp = self.network_connection.request('{}/{}/{}_router_interface'.format('/v2.0/routers', router.id, op), method='PUT', data=data)
        return resp.status == httplib.OK

    def ex_add_router_port(self, router, port):
        """
        Add port to a router

        :param router: Router to add the port
        :type router: :class:`OpenStack_2_Router`

        :param port: Port object to be added to the router
        :type port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        return self._manage_router_interface(router, 'add', port=port)

    def ex_del_router_port(self, router, port):
        """
        Remove port from a router

        :param router: Router to remove the port
        :type router: :class:`OpenStack_2_Router`

        :param port: Port object to be added to the router
        :type port: :class:`OpenStack_2_PortInterface`

        :rtype: ``bool``
        """
        return self._manage_router_interface(router, 'remove', port=port)

    def ex_add_router_subnet(self, router, subnet):
        """
        Add subnet to a router

        :param router: Router to add the subnet
        :type router: :class:`OpenStack_2_Router`

        :param subnet: Subnet object to be added to the router
        :type subnet: :class:`OpenStack_2_SubNet`

        :rtype: ``bool``
        """
        return self._manage_router_interface(router, 'add', subnet=subnet)

    def ex_del_router_subnet(self, router, subnet):
        """
        Remove subnet to a router

        :param router: Router to remove the subnet
        :type router: :class:`OpenStack_2_Router`

        :param subnet: Subnet object to be added to the router
        :type subnet: :class:`OpenStack_2_SubNet`

        :rtype: ``bool``
        """
        return self._manage_router_interface(router, 'remove', subnet=subnet)

    def _to_quota_set(self, obj):
        res = OpenStack_2_QuotaSet(id=obj['id'], cores=obj['cores'], instances=obj['instances'], key_pairs=obj['key_pairs'], metadata_items=obj['metadata_items'], ram=obj['ram'], server_groups=obj['server_groups'], server_group_members=obj['server_group_members'], fixed_ips=obj.get('fixed_ips', None), floating_ips=obj.get('floating_ips', None), networks=obj.get('networks', None), security_group_rules=obj.get('security_group_rules', None), security_groups=obj.get('security_groups', None), injected_file_content_bytes=obj.get('injected_file_content_bytes', None), injected_file_path_bytes=obj.get('injected_file_path_bytes', None), injected_files=obj.get('injected_files', None), driver=self.connection.driver)
        return res

    def ex_get_quota_set(self, tenant_id, user_id=None):
        """
        Get the quota for a project or a project and a user.

        :param      tenant_id: The UUID of the tenant in a multi-tenancy cloud
        :type       tenant_id: ``str``

        :param      user_id: ID of user to list the quotas for.
        :type       user_id: ``str``

        :rtype: :class:`OpenStack_2_QuotaSet`
        """
        url = '/os-quota-sets/%s/detail' % tenant_id
        if user_id:
            url += '?user_id=%s' % user_id
        return self._to_quota_set(self.connection.request(url).object['quota_set'])

    def _to_network_quota(self, obj):
        res = OpenStack_2_NetworkQuota(floatingip=obj['floatingip'], network=obj['network'], port=obj['port'], rbac_policy=obj['rbac_policy'], router=obj.get('router', None), security_group=obj.get('security_group', None), security_group_rule=obj.get('security_group_rule', None), subnet=obj.get('subnet', None), subnetpool=obj.get('subnetpool', None), driver=self.connection.driver)
        return res

    def ex_get_network_quotas(self, project_id):
        """
        Get the network quotas for a project

        :param      project_id: The ID of the project.
        :type       project_id: ``str``

        :rtype: :class:`OpenStack_2_NetworkQuota`
        """
        url = '/v2.0/quotas/%s/details.json' % project_id
        return self._to_network_quota(self.network_connection.request(url).object['quota'])

    def _to_volume_quota(self, obj):
        res = OpenStack_2_VolumeQuota(backup_gigabytes=obj.get('backup_gigabytes', None), gigabytes=obj.get('gigabytes', None), per_volume_gigabytes=obj.get('per_volume_gigabytes', None), backups=obj.get('backups', None), snapshots=obj.get('snapshots', None), volumes=obj.get('volumes', None), driver=self.connection.driver)
        return res

    def ex_get_volume_quotas(self, project_id):
        """
        Get the volume quotas for a project

        :param      project_id: The ID of the project.
        :type       project_id: ``str``

        :rtype: :class:`OpenStack_2_VolumeQuota`
        """
        url = '/os-quota-sets/%s?usage=True' % project_id
        return self._to_volume_quota(self._get_volume_connection().request(url).object['quota_set'])

    def ex_list_server_groups(self):
        """
        List Server Groups

        :rtype: ``list`` of :class:`OpenStack_2_ServerGroup`
        """
        return self._to_server_groups(self.connection.request('/os-server-groups').object)

    def _to_server_groups(self, obj):
        sg_elements = obj['server_groups']
        return [self._to_server_group(sg) for sg in sg_elements]

    def _to_server_group(self, obj):
        policy = None
        if 'policy' in obj:
            policy = obj['policy']
        elif 'policies' in obj and obj['policies']:
            policy = obj['policies'][0]
        return OpenStack_2_ServerGroup(id=obj['id'], name=obj['name'], policy=policy, members=obj.get('members'), rules=obj.get('rules'), driver=self.connection.driver)

    def ex_get_server_group(self, server_group_id):
        """
        Get Server Group

        :rtype: :class:`OpenStack_2_ServerGroup`
        """
        return self._to_server_group(self.connection.request('/os-server-groups/%s' % server_group_id).object['server_group'])

    def ex_add_server_group(self, name, policy, rules=[]):
        """
        Add a Server Group

        :param name: Server Group Name.
        :type name: ``str``
        :param policy: Server Group policy.
        :type policy: ``str``
        :param rules: Server Group rules.
        :type rules: ``list``

        :rtype: ``bool``
        """
        data = {'name': name}
        if rules:
            data['rules'] = rules
        try:
            data['policy'] = policy
            response = self.connection.request('/os-server-groups', method='POST', data={'server_group': data}).object
        except BaseHTTPError:
            del data['policy']
            data['policies'] = [policy]
            response = self.connection.request('/os-server-groups', method='POST', data={'server_group': data}).object
        return self._to_server_group(response['server_group'])

    def ex_del_server_group(self, server_group):
        """
        Delete a Server Group

        :param server_group: Server Group which should be deleted
        :type server_group: :class:`OpenStack_2_ServerGroup`

        :rtype: ``bool``
        """
        resp = self.connection.request('/os-server-groups/%s' % server_group.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def _to_floating_ips(self, obj):
        ip_elements = obj['floatingips']
        return [self._to_floating_ip(ip) for ip in ip_elements]

    def _to_floating_ip(self, obj):
        extra = {}
        extra['port_details'] = obj.get('port_details')
        extra['port_id'] = obj.get('port_id')
        extra['floating_network_id'] = obj.get('floating_network_id')
        return OpenStack_2_FloatingIpAddress(id=obj['id'], ip_address=obj['floating_ip_address'], pool=None, node_id=None, driver=self, extra=extra)

    def ex_list_floating_ips(self):
        """
        List floating IPs
        :rtype: ``list`` of :class:`OpenStack_2_FloatingIpAddress`
        """
        return self._to_floating_ips(self.network_connection.request('/v2.0/floatingips').object)

    def ex_get_floating_ip(self, ip):
        """
        Get specified floating IP from the pool
        :param      ip: floating IP to get
        :type       ip: ``str``
        :rtype: :class:`OpenStack_2_FloatingIpAddress`
        """
        floating_ips = self._to_floating_ips(self.network_connection.request('/v2.0/floatingips?floating_ip_address=%s' % ip).object)
        return floating_ips[0] if floating_ips else None

    def ex_create_floating_ip(self, ip_pool):
        """
        Create new floating IP. The ip_pool attribute is optional only if your
        infrastructure has only one IP pool available.
        :param      ip_pool: name or id of the floating IP pool
        :type       ip_pool: ``str``
        :rtype: :class:`OpenStack_2_FloatingIpAddress`
        """
        for pool in self.ex_list_floating_ip_pools():
            if not ip_pool or ip_pool == pool.name or ip_pool == pool.id:
                return pool.create_floating_ip()

    def ex_delete_floating_ip(self, ip):
        """
        Delete specified floating IP
        :param      ip: floating IP to remove
        :type       ip: :class:`OpenStack_2_FloatingIpAddress`
        :rtype: ``bool``
        """
        resp = self.network_connection.request('/v2.0/floatingips/%s' % ip.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def ex_attach_floating_ip_to_node(self, node, ip):
        """
        Attach the floating IP to the node

        :param      node: node
        :type       node: :class:`Node`

        :param      ip: floating IP to attach
        :type       ip: ``str`` or :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        ip_id = None
        if hasattr(ip, 'id'):
            ip_id = ip.id
        else:
            for pool in self.ex_list_floating_ip_pools():
                fip = pool.get_floating_ip(ip)
                if fip:
                    ip_id = fip.id
        if not ip_id:
            return False
        ports = self.ex_get_node_ports(node)
        if ports:
            resp = self.network_connection.request('/v2.0/floatingips/%s' % ip_id, method='PUT', data={'floatingip': {'port_id': ports[0].id}})
            return resp.status == httplib.OK
        else:
            return False

    def ex_detach_floating_ip_from_node(self, node, ip):
        """
        Detach the floating IP from the node

        :param      node: node
        :type       node: :class:`Node`

        :param      ip: floating IP to remove
        :type       ip: ``str`` or :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        ip_id = None
        if hasattr(ip, 'id'):
            ip_id = ip.id
        else:
            for pool in self.ex_list_floating_ip_pools():
                fip = pool.get_floating_ip(ip)
                if fip:
                    ip_id = fip.id
        if not ip_id:
            return False
        resp = self.network_connection.request('/v2.0/floatingips/%s' % ip_id, method='PUT', data={'floatingip': {'port_id': None}})
        return resp.status == httplib.OK