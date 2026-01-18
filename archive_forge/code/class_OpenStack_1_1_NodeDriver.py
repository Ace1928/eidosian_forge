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
class OpenStack_1_1_NodeDriver(OpenStackNodeDriver):
    """
    OpenStack node driver.
    """
    connectionCls = OpenStack_1_1_Connection
    type = Provider.OPENSTACK
    features = {'create_node': ['generates_password']}
    _networks_url_prefix = '/os-networks'

    def __init__(self, *args, **kwargs):
        self._ex_force_api_version = str(kwargs.pop('ex_force_api_version', None))
        super().__init__(*args, **kwargs)

    def create_node(self, name, size, image=None, ex_keyname=None, ex_userdata=None, ex_config_drive=None, ex_security_groups=None, ex_metadata=None, ex_files=None, networks=None, ex_disk_config=None, ex_admin_pass=None, ex_availability_zone=None, ex_blockdevicemappings=None, ex_os_scheduler_hints=None):
        """Create a new node

        @inherits:  :class:`NodeDriver.create_node`

        :keyword    ex_keyname:  The name of the key pair
        :type       ex_keyname:  ``str``

        :keyword    ex_userdata: String containing user data
                                 see
                                 https://help.ubuntu.com/community/CloudInit
        :type       ex_userdata: ``str``

        :keyword    ex_config_drive: Enable config drive
                                     see
                                     http://docs.openstack.org/grizzly/openstack-compute/admin/content/config-drive.html
        :type       ex_config_drive: ``bool``

        :keyword    ex_security_groups: List of security groups to assign to
                                        the node
        :type       ex_security_groups: ``list`` of
                                       :class:`OpenStackSecurityGroup`

        :keyword    ex_metadata: Key/Value metadata to associate with a node
        :type       ex_metadata: ``dict``

        :keyword    ex_files:   File Path => File contents to create on
                                the node
        :type       ex_files:   ``dict``


        :keyword    networks: The server is launched into a set of Networks.
        :type       networks: ``list`` of :class:`OpenStackNetwork`

        :keyword    ex_disk_config: Name of the disk configuration.
                                    Can be either ``AUTO`` or ``MANUAL``.
        :type       ex_disk_config: ``str``

        :keyword    ex_config_drive: If True enables metadata injection in a
                                     server through a configuration drive.
        :type       ex_config_drive: ``bool``

        :keyword    ex_admin_pass: The root password for the node
        :type       ex_admin_pass: ``str``

        :keyword    ex_availability_zone: Nova availability zone for the node
        :type       ex_availability_zone: ``str``

        :keyword    ex_blockdevicemappings: Enables fine grained control of the
                                            block device mapping for an instance.
        :type       ex_blockdevicemappings: ``dict``

        :keyword    ex_os_scheduler_hints: The dictionary of data to send to
                                           the scheduler.
        :type       ex_os_scheduler_hints:   ``dict``
        """
        ex_metadata = ex_metadata or {}
        ex_files = ex_files or {}
        networks = networks or []
        ex_security_groups = ex_security_groups or []
        server_params = self._create_args_to_params(node=None, name=name, size=size, image=image, ex_keyname=ex_keyname, ex_userdata=ex_userdata, ex_config_drive=ex_config_drive, ex_security_groups=ex_security_groups, ex_metadata=ex_metadata, ex_files=ex_files, networks=networks, ex_disk_config=ex_disk_config, ex_availability_zone=ex_availability_zone, ex_blockdevicemappings=ex_blockdevicemappings)
        data = {'server': server_params}
        if ex_os_scheduler_hints:
            data['os:scheduler_hints'] = ex_os_scheduler_hints
        resp = self.connection.request('/servers', method='POST', data=data)
        create_response = resp.object['server']
        server_resp = self.connection.request('/servers/%s' % create_response['id'])
        server_object = server_resp.object['server']
        server_object['adminPass'] = create_response.get('adminPass', None)
        return self._to_node(server_object)

    def _to_images(self, obj, ex_only_active):
        images = []
        for image in obj['images']:
            if ex_only_active and image.get('status') != 'ACTIVE':
                continue
            images.append(self._to_image(image))
        return images

    def _to_image(self, api_image):
        server = api_image.get('server', {})
        updated = api_image.get('updated_at') or api_image['updated']
        created = api_image.get('created_at') or api_image['created']
        min_ram = api_image.get('min_ram')
        if min_ram is None:
            min_ram = api_image.get('minRam')
        min_disk = api_image.get('min_disk')
        if min_disk is None:
            min_disk = api_image.get('minDisk')
        return NodeImage(id=api_image['id'], name=api_image['name'], driver=self, extra=dict(visibility=api_image.get('visibility'), updated=updated, created=created, status=api_image['status'], progress=api_image.get('progress'), metadata=api_image.get('metadata'), os_type=api_image.get('os_type'), serverId=server.get('id'), minDisk=min_disk, minRam=min_ram))

    def _to_image_member(self, api_image_member):
        created = api_image_member['created_at']
        updated = api_image_member.get('updated_at')
        return NodeImageMember(id=api_image_member['member_id'], image_id=api_image_member['image_id'], state=api_image_member['status'], created=created, driver=self, extra=dict(schema=api_image_member.get('schema'), updated=updated))

    def _to_nodes(self, obj):
        servers = obj['servers']
        return [self._to_node(server) for server in servers]

    def _to_volumes(self, obj):
        volumes = obj['volumes']
        return [self._to_volume(volume) for volume in volumes]

    def _to_snapshots(self, obj):
        snapshots = obj['snapshots']
        return [self._to_snapshot(snapshot) for snapshot in snapshots]

    def _to_sizes(self, obj):
        flavors = obj['flavors']
        return [self._to_size(flavor) for flavor in flavors]

    def _create_args_to_params(self, node, **kwargs):
        server_params = {'name': kwargs.get('name'), 'metadata': kwargs.get('ex_metadata', {}) or {}}
        if kwargs.get('ex_files', None):
            server_params['personality'] = self._files_to_personality(kwargs.get('ex_files'))
        if kwargs.get('ex_availability_zone', None):
            server_params['availability_zone'] = kwargs['ex_availability_zone']
        if kwargs.get('ex_keyname', None):
            server_params['key_name'] = kwargs['ex_keyname']
        if kwargs.get('ex_userdata', None):
            server_params['user_data'] = base64.b64encode(b(kwargs['ex_userdata'])).decode('ascii')
        if kwargs.get('ex_disk_config', None):
            server_params['OS-DCF:diskConfig'] = kwargs['ex_disk_config']
        if kwargs.get('ex_config_drive', None):
            server_params['config_drive'] = str(kwargs['ex_config_drive'])
        if kwargs.get('ex_admin_pass', None):
            server_params['adminPass'] = kwargs['ex_admin_pass']
        if kwargs.get('networks', None):
            networks = kwargs['networks'] or []
            networks = [{'uuid': network.id} for network in networks]
            server_params['networks'] = networks
        if kwargs.get('ex_security_groups', None):
            server_params['security_groups'] = []
            for security_group in kwargs['ex_security_groups'] or []:
                name = security_group.name
                server_params['security_groups'].append({'name': name})
        if kwargs.get('ex_blockdevicemappings', None):
            server_params['block_device_mapping_v2'] = kwargs['ex_blockdevicemappings']
        if kwargs.get('name', None):
            server_params['name'] = kwargs.get('name')
        else:
            server_params['name'] = node.name
        if kwargs.get('image', None):
            server_params['imageRef'] = kwargs.get('image').id
        else:
            server_params['imageRef'] = node.extra.get('imageId', '') if node else ''
        if kwargs.get('size', None):
            server_params['flavorRef'] = kwargs.get('size').id
        else:
            server_params['flavorRef'] = node.extra.get('flavorId')
        return server_params

    def _files_to_personality(self, files):
        rv = []
        for k, v in list(files.items()):
            rv.append({'path': k, 'contents': base64.b64encode(b(v)).decode()})
        return rv

    def _reboot_node(self, node, reboot_type='SOFT'):
        resp = self._node_action(node, 'reboot', type=reboot_type)
        return resp.status == httplib.ACCEPTED

    def ex_set_password(self, node, password):
        """
        Changes the administrator password for a specified server.

        :param      node: Node to rebuild.
        :type       node: :class:`Node`

        :param      password: The administrator password.
        :type       password: ``str``

        :rtype: ``bool``
        """
        resp = self._node_action(node, 'changePassword', adminPass=password)
        node.extra['password'] = password
        return resp.status == httplib.ACCEPTED

    def ex_rebuild(self, node, image, **kwargs):
        """
        Rebuild a Node.

        :param      node: Node to rebuild.
        :type       node: :class:`Node`

        :param      image: New image to use.
        :type       image: :class:`NodeImage`

        :keyword    ex_metadata: Key/Value metadata to associate with a node
        :type       ex_metadata: ``dict``

        :keyword    ex_files:   File Path => File contents to create on
                                the node
        :type       ex_files:   ``dict``

        :keyword    ex_keyname:  Name of existing public key to inject into
                                 instance
        :type       ex_keyname:  ``str``

        :keyword    ex_userdata: String containing user data
                                 see
                                 https://help.ubuntu.com/community/CloudInit
        :type       ex_userdata: ``str``

        :keyword    ex_security_groups: List of security groups to assign to
                                        the node
        :type       ex_security_groups: ``list`` of
                                       :class:`OpenStackSecurityGroup`

        :keyword    ex_disk_config: Name of the disk configuration.
                                    Can be either ``AUTO`` or ``MANUAL``.
        :type       ex_disk_config: ``str``

        :keyword    ex_config_drive: If True enables metadata injection in a
                                     server through a configuration drive.
        :type       ex_config_drive: ``bool``

        :rtype: ``bool``
        """
        server_params = self._create_args_to_params(node, image=image, **kwargs)
        resp = self._node_action(node, 'rebuild', **server_params)
        return resp.status == httplib.ACCEPTED

    def ex_resize(self, node, size):
        """
        Change a node size.

        :param      node: Node to resize.
        :type       node: :class:`Node`

        :type       size: :class:`NodeSize`
        :param      size: New size to use.

        :rtype: ``bool``
        """
        server_params = {'flavorRef': size.id}
        resp = self._node_action(node, 'resize', **server_params)
        return resp.status == httplib.ACCEPTED

    def ex_confirm_resize(self, node):
        """
        Confirms a pending resize action.

        :param      node: Node to resize.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        resp = self._node_action(node, 'confirmResize')
        return resp.status == httplib.NO_CONTENT

    def ex_revert_resize(self, node):
        """
        Cancels and reverts a pending resize action.

        :param      node: Node to resize.
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        resp = self._node_action(node, 'revertResize')
        return resp.status == httplib.ACCEPTED

    def create_image(self, node, name, metadata=None):
        """
        Creates a new image.

        :param      node: Node
        :type       node: :class:`Node`

        :param      name: The name for the new image.
        :type       name: ``str``

        :param      metadata: Key and value pairs for metadata.
        :type       metadata: ``dict``

        :rtype: :class:`NodeImage`
        """
        optional_params = {}
        if metadata:
            optional_params['metadata'] = metadata
        resp = self._node_action(node, 'createImage', name=name, **optional_params)
        image_id = self._extract_image_id_from_url(resp.headers['location'])
        return self.get_image(image_id=image_id)

    def ex_set_server_name(self, node, name):
        """
        Sets the Node's name.

        :param      node: Node
        :type       node: :class:`Node`

        :param      name: The name of the server.
        :type       name: ``str``

        :rtype: :class:`Node`
        """
        return self._update_node(node, name=name)

    def ex_get_metadata(self, node):
        """
        Get a Node's metadata.

        :param      node: Node
        :type       node: :class:`Node`

        :return: Key/Value metadata associated with node.
        :rtype: ``dict``
        """
        return self.connection.request('/servers/{}/metadata'.format(node.id), method='GET').object['metadata']

    def ex_set_metadata(self, node, metadata):
        """
        Sets the Node's metadata.

        :param      node: Node
        :type       node: :class:`Node`

        :param      metadata: Key/Value metadata to associate with a node
        :type       metadata: ``dict``

        :rtype: ``dict``
        """
        return self.connection.request('/servers/{}/metadata'.format(node.id), method='PUT', data={'metadata': metadata}).object['metadata']

    def ex_update_node(self, node, **node_updates):
        """
        Update the Node's editable attributes.  The OpenStack API currently
        supports editing name and IPv4/IPv6 access addresses.

        The driver currently only supports updating the node name.

        :param      node: Node
        :type       node: :class:`Node`

        :keyword    name:   New name for the server
        :type       name:   ``str``

        :rtype: :class:`Node`
        """
        potential_data = self._create_args_to_params(node, **node_updates)
        updates = {'name': potential_data['name']}
        return self._update_node(node, **updates)

    def _to_networks(self, obj):
        networks = obj['networks']
        return [self._to_network(network) for network in networks]

    def _to_network(self, obj):
        return OpenStackNetwork(id=obj['id'], name=obj['label'], cidr=obj.get('cidr', None), driver=self)

    def ex_list_networks(self):
        """
        Get a list of Networks that are available.

        :rtype: ``list`` of :class:`OpenStackNetwork`
        """
        response = self.connection.request(self._networks_url_prefix).object
        return self._to_networks(response)

    def ex_get_network(self, network_id):
        """
        Retrieve the Network with the given ID

        :param networkId: ID of the network
        :type networkId: ``str``

        :rtype :class:`OpenStackNetwork`
        """
        request_url = '{networks_url_prefix}/{network_id}'.format(networks_url_prefix=self._networks_url_prefix, network_id=network_id)
        response = self.connection.request(request_url).object
        return self._to_network(response['network'])

    def ex_create_network(self, name, cidr):
        """
        Create a new Network

        :param name: Name of network which should be used
        :type name: ``str``

        :param cidr: cidr of network which should be used
        :type cidr: ``str``

        :rtype: :class:`OpenStackNetwork`
        """
        data = {'network': {'cidr': cidr, 'label': name}}
        response = self.connection.request(self._networks_url_prefix, method='POST', data=data).object
        return self._to_network(response['network'])

    def ex_delete_network(self, network):
        """
        Delete a Network

        :param network: Network which should be used
        :type network: :class:`OpenStackNetwork`

        :rtype: ``bool``
        """
        resp = self.connection.request('{}/{}'.format(self._networks_url_prefix, network.id), method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def ex_get_console_output(self, node, length=None):
        """
        Get console output

        :param      node: node
        :type       node: :class:`Node`

        :param      length: Optional number of lines to fetch from the
                            console log
        :type       length: ``int``

        :return: Dictionary with the output
        :rtype: ``dict``
        """
        data = {'os-getConsoleOutput': {'length': length}}
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=data).object
        return resp

    def ex_list_snapshots(self):
        return self._to_snapshots(self.connection.request('/os-snapshots').object)

    def ex_get_snapshot(self, snapshotId):
        return self._to_snapshot(self.connection.request('/os-snapshots/%s' % snapshotId).object)

    def list_volume_snapshots(self, volume):
        return [snapshot for snapshot in self.ex_list_snapshots() if snapshot.extra['volume_id'] == volume.id]

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
            data['snapshot']['display_name'] = name
        if ex_description is not None:
            data['snapshot']['display_description'] = ex_description
        return self._to_snapshot(self.connection.request('/os-snapshots', method='POST', data=data).object)

    def destroy_volume_snapshot(self, snapshot):
        resp = self.connection.request('/os-snapshots/%s' % snapshot.id, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def ex_create_snapshot(self, volume, name, description=None, force=False):
        """
        Create a snapshot based off of a volume.

        :param      volume: volume
        :type       volume: :class:`StorageVolume`

        :keyword    name: New name for the volume snapshot
        :type       name: ``str``

        :keyword    description: Description of the snapshot (optional)
        :type       description: ``str``

        :keyword    force: Whether to force creation (optional)
        :type       force: ``bool``

        :rtype:     :class:`VolumeSnapshot`
        """
        warnings.warn('This method has been deprecated in favor of the create_volume_snapshot method')
        return self.create_volume_snapshot(volume, name, ex_description=description, ex_force=force)

    def ex_delete_snapshot(self, snapshot):
        """
        Delete a VolumeSnapshot

        :param      snapshot: snapshot
        :type       snapshot: :class:`VolumeSnapshot`

        :rtype:     ``bool``
        """
        warnings.warn('This method has been deprecated in favor of the destroy_volume_snapshot method')
        return self.destroy_volume_snapshot(snapshot)

    def _to_security_group_rules(self, obj):
        return [self._to_security_group_rule(security_group_rule) for security_group_rule in obj]

    def _to_security_group_rule(self, obj):
        ip_range = group = tenant_id = None
        if obj['group'] == {}:
            ip_range = obj['ip_range'].get('cidr', None)
        else:
            group = obj['group'].get('name', None)
            tenant_id = obj['group'].get('tenant_id', None)
        return OpenStackSecurityGroupRule(id=obj['id'], parent_group_id=obj['parent_group_id'], ip_protocol=obj['ip_protocol'], from_port=obj['from_port'], to_port=obj['to_port'], driver=self, ip_range=ip_range, group=group, tenant_id=tenant_id)

    def _to_security_groups(self, obj):
        security_groups = obj['security_groups']
        return [self._to_security_group(security_group) for security_group in security_groups]

    def _to_security_group(self, obj):
        rules = self._to_security_group_rules(obj.get('security_group_rules', obj.get('rules', [])))
        return OpenStackSecurityGroup(id=obj['id'], tenant_id=obj['tenant_id'], name=obj['name'], description=obj.get('description', ''), rules=rules, driver=self)

    def ex_list_security_groups(self):
        """
        Get a list of Security Groups that are available.

        :rtype: ``list`` of :class:`OpenStackSecurityGroup`
        """
        return self._to_security_groups(self.connection.request('/os-security-groups').object)

    def ex_get_node_security_groups(self, node):
        """
        Get Security Groups of the specified server.

        :rtype: ``list`` of :class:`OpenStackSecurityGroup`
        """
        return self._to_security_groups(self.connection.request('/servers/%s/os-security-groups' % node.id).object)

    def ex_create_security_group(self, name, description):
        """
        Create a new Security Group

        :param name: Name of the new Security Group
        :type  name: ``str``

        :param description: Description of the new Security Group
        :type  description: ``str``

        :rtype: :class:`OpenStackSecurityGroup`
        """
        return self._to_security_group(self.connection.request('/os-security-groups', method='POST', data={'security_group': {'name': name, 'description': description}}).object['security_group'])

    def ex_delete_security_group(self, security_group):
        """
        Delete a Security Group.

        :param security_group: Security Group should be deleted
        :type  security_group: :class:`OpenStackSecurityGroup`

        :rtype: ``bool``
        """
        resp = self.connection.request('/os-security-groups/%s' % security_group.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

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
        return self._to_security_group_rule(self.connection.request('/os-security-group-rules', method='POST', data={'security_group_rule': {'ip_protocol': ip_protocol, 'from_port': from_port, 'to_port': to_port, 'cidr': cidr, 'group_id': source_security_group_id, 'parent_group_id': security_group.id}}).object['security_group_rule'])

    def ex_delete_security_group_rule(self, rule):
        """
        Delete a Rule from a Security Group.

        :param rule: Rule should be deleted
        :type  rule: :class:`OpenStackSecurityGroupRule`

        :rtype: ``bool``
        """
        resp = self.connection.request('/os-security-group-rules/%s' % rule.id, method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def _to_key_pairs(self, obj):
        key_pairs = obj['keypairs']
        key_pairs = [self._to_key_pair(key_pair['keypair']) for key_pair in key_pairs]
        return key_pairs

    def _to_key_pair(self, obj):
        key_pair = KeyPair(name=obj['name'], fingerprint=obj['fingerprint'], public_key=obj['public_key'], private_key=obj.get('private_key', None), driver=self)
        return key_pair

    def list_key_pairs(self):
        response = self.connection.request('/os-keypairs')
        key_pairs = self._to_key_pairs(response.object)
        return key_pairs

    def get_key_pair(self, name):
        self.connection.set_context({'key_pair_name': name})
        response = self.connection.request('/os-keypairs/%s' % name)
        key_pair = self._to_key_pair(response.object['keypair'])
        return key_pair

    def create_key_pair(self, name):
        data = {'keypair': {'name': name}}
        response = self.connection.request('/os-keypairs', method='POST', data=data)
        key_pair = self._to_key_pair(response.object['keypair'])
        return key_pair

    def import_key_pair_from_string(self, name, key_material):
        data = {'keypair': {'name': name, 'public_key': key_material}}
        response = self.connection.request('/os-keypairs', method='POST', data=data)
        key_pair = self._to_key_pair(response.object['keypair'])
        return key_pair

    def delete_key_pair(self, key_pair):
        """
        Delete a KeyPair.

        :param keypair: KeyPair to delete
        :type  keypair: :class:`OpenStackKeyPair`

        :rtype: ``bool``
        """
        response = self.connection.request('/os-keypairs/%s' % key_pair.name, method='DELETE')
        return response.status == httplib.ACCEPTED

    def ex_list_keypairs(self):
        """
        Get a list of KeyPairs that are available.

        :rtype: ``list`` of :class:`OpenStackKeyPair`
        """
        warnings.warn('This method has been deprecated in favor of list_key_pairs method')
        return self.list_key_pairs()

    def ex_create_keypair(self, name):
        """
        Create a new KeyPair

        :param name: Name of the new KeyPair
        :type  name: ``str``

        :rtype: :class:`OpenStackKeyPair`
        """
        warnings.warn('This method has been deprecated in favor of create_key_pair method')
        return self.create_key_pair(name=name)

    def ex_import_keypair(self, name, keyfile):
        """
        Import a KeyPair from a file

        :param name: Name of the new KeyPair
        :type  name: ``str``

        :param keyfile: Path to the public key file (in OpenSSH format)
        :type  keyfile: ``str``

        :rtype: :class:`OpenStackKeyPair`
        """
        warnings.warn('This method has been deprecated in favor of import_key_pair_from_file method')
        return self.import_key_pair_from_file(name=name, key_file_path=keyfile)

    def ex_import_keypair_from_string(self, name, key_material):
        """
        Import a KeyPair from a string

        :param name: Name of the new KeyPair
        :type  name: ``str``

        :param key_material: Public key (in OpenSSH format)
        :type  key_material: ``str``

        :rtype: :class:`OpenStackKeyPair`
        """
        warnings.warn('This method has been deprecated in favor of import_key_pair_from_string method')
        return self.import_key_pair_from_string(name=name, key_material=key_material)

    def ex_delete_keypair(self, keypair):
        """
        Delete a KeyPair.

        :param keypair: KeyPair to delete
        :type  keypair: :class:`OpenStackKeyPair`

        :rtype: ``bool``
        """
        warnings.warn('This method has been deprecated in favor of delete_key_pair method')
        return self.delete_key_pair(key_pair=keypair)

    def ex_get_size(self, size_id):
        """
        Get a NodeSize

        :param      size_id: ID of the size which should be used
        :type       size_id: ``str``

        :rtype: :class:`NodeSize`
        """
        return self._to_size(self.connection.request('/flavors/{}'.format(size_id)).object['flavor'])

    def ex_get_size_extra_specs(self, size_id):
        """
        Get the extra_specs field of a NodeSize

        :param      size_id: ID of the size which should be used
        :type       size_id: ``str``

        :rtype: `dict`
        """
        return self.connection.request('/flavors/{}/os-extra_specs'.format(size_id)).object['extra_specs']

    def get_image(self, image_id):
        """
        Get a NodeImage

        @inherits: :class:`NodeDriver.get_image`

        :param      image_id: ID of the image which should be used
        :type       image_id: ``str``

        :rtype: :class:`NodeImage`
        """
        return self._to_image(self.connection.request('/images/{}'.format(image_id)).object['image'])

    def delete_image(self, image):
        """
        Delete a NodeImage

        @inherits: :class:`NodeDriver.delete_image`

        :param      image: image witch should be used
        :type       image: :class:`NodeImage`

        :rtype: ``bool``
        """
        resp = self.connection.request('/images/{}'.format(image.id), method='DELETE')
        return resp.status == httplib.NO_CONTENT

    def _node_action(self, node, action, **params):
        params = params or None
        return self.connection.request('/servers/{}/action'.format(node.id), method='POST', data={action: params})

    def _update_node(self, node, **node_updates):
        """
        Updates the editable attributes of a server, which currently include
        its name and IPv4/IPv6 access addresses.
        """
        return self._to_node(self.connection.request('/servers/{}'.format(node.id), method='PUT', data={'server': node_updates}).object['server'])

    def _to_node_from_obj(self, obj):
        return self._to_node(obj['server'])

    def _to_node(self, api_node):
        public_networks_labels = ['public', 'internet']
        public_ips, private_ips = ([], [])
        for label, values in api_node['addresses'].items():
            for value in values:
                ip = value['addr']
                is_public_ip = False
                try:
                    is_public_ip = is_public_subnet(ip)
                except Exception:
                    explicit_ip_type = value.get('OS-EXT-IPS:type', None)
                    if label in public_networks_labels:
                        is_public_ip = True
                    elif explicit_ip_type == 'floating':
                        is_public_ip = True
                    elif explicit_ip_type == 'fixed':
                        is_public_ip = False
                if is_public_ip:
                    public_ips.append(ip)
                else:
                    private_ips.append(ip)
        image = api_node.get('image', None)
        image_id = image.get('id', None) if image else None
        config_drive = api_node.get('config_drive', False)
        volumes_attached = api_node.get('os-extended-volumes:volumes_attached')
        created = parse_date(api_node['created'])
        return Node(id=api_node['id'], name=api_node['name'], state=self.NODE_STATE_MAP.get(api_node['status'], NodeState.UNKNOWN), public_ips=public_ips, private_ips=private_ips, created_at=created, driver=self, extra=dict(addresses=api_node['addresses'], hostId=api_node['hostId'], access_ip=api_node.get('accessIPv4'), access_ipv6=api_node.get('accessIPv6', None), tenantId=api_node.get('tenant_id') or api_node['tenantId'], userId=api_node.get('user_id', None), imageId=image_id, flavorId=api_node.get('flavor', {}).get('id', None), flavor_details=api_node.get('flavor', None), uri=next((link['href'] for link in api_node['links'] if link['rel'] == 'self')), service_name=self.connection.get_service_name(), metadata=api_node['metadata'], password=api_node.get('adminPass', None), created=api_node['created'], updated=api_node['updated'], key_name=api_node.get('key_name', None), disk_config=api_node.get('OS-DCF:diskConfig', None), config_drive=config_drive, availability_zone=api_node.get('OS-EXT-AZ:availability_zone'), volumes_attached=volumes_attached, task_state=api_node.get('OS-EXT-STS:task_state', None), vm_state=api_node.get('OS-EXT-STS:vm_state', None), power_state=api_node.get('OS-EXT-STS:power_state', None), progress=api_node.get('progress', None), fault=api_node.get('fault')))

    def _to_volume(self, api_node):
        if 'volume' in api_node:
            api_node = api_node['volume']
        state = self.VOLUME_STATE_MAP.get(api_node['status'], StorageVolumeState.UNKNOWN)
        return StorageVolume(id=api_node['id'], name=api_node.get('displayName', api_node.get('name')), size=api_node['size'], state=state, driver=self, extra={'description': api_node.get('displayDescription', api_node.get('description')), 'attachments': [att for att in api_node['attachments'] if att], 'state': api_node.get('status', None), 'snapshot_id': api_node.get('snapshot_id', api_node.get('snapshotId')), 'location': api_node.get('availability_zone', api_node.get('availabilityZone')), 'volume_type': api_node.get('volume_type', api_node.get('volumeType')), 'metadata': api_node.get('metadata', None), 'created_at': api_node.get('created_at', api_node.get('createdAt'))})

    def _to_snapshot(self, data):
        if 'snapshot' in data:
            data = data['snapshot']
        volume_id = data.get('volume_id', data.get('volumeId', None))
        display_name = data.get('name', data.get('display_name', data.get('displayName', None)))
        created_at = data.get('created_at', data.get('createdAt', None))
        description = data.get('description', data.get('display_description', data.get('displayDescription', None)))
        status = data.get('status', None)
        extra = {'volume_id': volume_id, 'name': display_name, 'created': created_at, 'description': description, 'status': status}
        state = self.SNAPSHOT_STATE_MAP.get(status, VolumeSnapshotState.UNKNOWN)
        try:
            created_dt = parse_date(created_at)
        except ValueError:
            created_dt = None
        snapshot = VolumeSnapshot(id=data['id'], driver=self, size=data['size'], extra=extra, created=created_dt, state=state, name=display_name)
        return snapshot

    def _to_size(self, api_flavor, price=None, bandwidth=None):
        if not price:
            price = self._get_size_price(str(api_flavor['id']))
        extra = api_flavor.get('OS-FLV-WITH-EXT-SPECS:extra_specs', {})
        extra['disabled'] = api_flavor.get('OS-FLV-DISABLED:disabled', None)
        return OpenStackNodeSize(id=api_flavor['id'], name=api_flavor['name'], ram=api_flavor['ram'], disk=api_flavor['disk'], vcpus=api_flavor['vcpus'], ephemeral_disk=api_flavor.get('OS-FLV-EXT-DATA:ephemeral', None), swap=api_flavor['swap'], extra=extra, bandwidth=bandwidth, price=price, driver=self)

    def _get_size_price(self, size_id):
        try:
            return get_size_price(driver_type='compute', driver_name=self.api_name, size_id=size_id)
        except KeyError:
            return 0.0

    def _extract_image_id_from_url(self, location_header):
        path = urlparse.urlparse(location_header).path
        image_id = path.split('/')[-1]
        return image_id

    def ex_rescue(self, node, password=None):
        """
        Rescue a node

        :param      node: node
        :type       node: :class:`Node`

        :param      password: password
        :type       password: ``str``

        :rtype: :class:`Node`
        """
        if password:
            resp = self._node_action(node, 'rescue', adminPass=password)
        else:
            resp = self._node_action(node, 'rescue')
            password = json.loads(resp.body)['adminPass']
        node.extra['password'] = password
        return node

    def ex_unrescue(self, node):
        """
        Unrescue a node

        :param      node: node
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        resp = self._node_action(node, 'unrescue')
        return resp.status == httplib.ACCEPTED

    def _to_floating_ip_pools(self, obj):
        pool_elements = obj['floating_ip_pools']
        return [self._to_floating_ip_pool(pool) for pool in pool_elements]

    def _to_floating_ip_pool(self, obj):
        return OpenStack_1_1_FloatingIpPool(obj['name'], self.connection)

    def ex_list_floating_ip_pools(self):
        """
        List available floating IP pools

        :rtype: ``list`` of :class:`OpenStack_1_1_FloatingIpPool`
        """
        return self._to_floating_ip_pools(self.connection.request('/os-floating-ip-pools').object)

    def _to_floating_ips(self, obj):
        ip_elements = obj['floating_ips']
        return [self._to_floating_ip(ip) for ip in ip_elements]

    def _to_floating_ip(self, obj):
        return OpenStack_1_1_FloatingIpAddress(id=obj['id'], ip_address=obj['ip'], pool=None, node_id=obj['instance_id'], driver=self)

    def ex_list_floating_ips(self):
        """
        List floating IPs

        :rtype: ``list`` of :class:`OpenStack_1_1_FloatingIpAddress`
        """
        return self._to_floating_ips(self.connection.request('/os-floating-ips').object)

    def ex_get_floating_ip(self, ip):
        """
        Get specified floating IP

        :param      ip: floating IP to get
        :type       ip: ``str``

        :rtype: :class:`OpenStack_1_1_FloatingIpAddress`
        """
        for floating_ip in self.ex_list_floating_ips():
            if floating_ip.ip_address == ip:
                return floating_ip
        return None

    def ex_create_floating_ip(self, ip_pool=None):
        """
        Create new floating IP. The ip_pool attribute is optional only if your
        infrastructure has only one IP pool available.

        :param      ip_pool: name of the floating IP pool
        :type       ip_pool: ``str``

        :rtype: :class:`OpenStack_1_1_FloatingIpAddress`
        """
        data = {'pool': ip_pool} if ip_pool is not None else {}
        resp = self.connection.request('/os-floating-ips', method='POST', data=data)
        data = resp.object['floating_ip']
        id = data['id']
        ip_address = data['ip']
        return OpenStack_1_1_FloatingIpAddress(id=id, ip_address=ip_address, pool=None, node_id=None, driver=self)

    def ex_delete_floating_ip(self, ip):
        """
        Delete specified floating IP

        :param      ip: floating IP to remove
        :type       ip: :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        resp = self.connection.request('/os-floating-ips/%s' % ip.id, method='DELETE')
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
        address = ip.ip_address if hasattr(ip, 'ip_address') else ip
        data = {'addFloatingIp': {'address': address}}
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=data)
        return resp.status == httplib.ACCEPTED

    def ex_detach_floating_ip_from_node(self, node, ip):
        """
        Detach the floating IP from the node

        :param      node: node
        :type       node: :class:`Node`

        :param      ip: floating IP to remove
        :type       ip: ``str`` or :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        address = ip.ip_address if hasattr(ip, 'ip_address') else ip
        data = {'removeFloatingIp': {'address': address}}
        resp = self.connection.request('/servers/%s/action' % node.id, method='POST', data=data)
        return resp.status == httplib.ACCEPTED

    def ex_get_metadata_for_node(self, node):
        """
        Return the metadata associated with the node.

        :param      node: Node instance
        :type       node: :class:`Node`

        :return: A dictionary or other mapping of strings to strings,
                 associating tag names with tag values.
        :type tags: ``dict``
        """
        return node.extra['metadata']

    def ex_pause_node(self, node):
        return self._post_simple_node_action(node, 'pause')

    def ex_unpause_node(self, node):
        return self._post_simple_node_action(node, 'unpause')

    def ex_start_node(self, node):
        return self.start_node(node=node)

    def ex_stop_node(self, node):
        return self.stop_node(node=node)

    def ex_suspend_node(self, node):
        return self._post_simple_node_action(node, 'suspend')

    def ex_resume_node(self, node):
        return self._post_simple_node_action(node, 'resume')

    def _post_simple_node_action(self, node, action):
        """Post a simple, data-less action to the OS node action endpoint
        :param `Node` node:
        :param str action: the action to call
        :return `bool`: a boolean that indicates success
        """
        uri = '/servers/{node_id}/action'.format(node_id=node.id)
        resp = self.connection.request(uri, method='POST', data={action: None})
        return resp.status == httplib.ACCEPTED