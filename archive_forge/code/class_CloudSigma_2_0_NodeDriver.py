import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigma_2_0_NodeDriver(CloudSigmaNodeDriver):
    """
    Driver for CloudSigma API v2.0.
    """
    name = 'CloudSigma (API v2.0)'
    api_name = 'cloudsigma_zrh'
    website = 'http://www.cloudsigma.com/'
    connectionCls = CloudSigma_2_0_Connection
    DRIVE_TRANSITION_TIMEOUT = 500
    DRIVE_TRANSITION_SLEEP_INTERVAL = 5
    NODE_STATE_MAP = {'starting': NodeState.PENDING, 'stopping': NodeState.PENDING, 'unavailable': NodeState.ERROR, 'running': NodeState.RUNNING, 'stopped': NodeState.STOPPED, 'paused': NodeState.PAUSED}

    def __init__(self, key, secret, secure=True, host=None, port=None, region=DEFAULT_REGION, **kwargs):
        if region not in API_ENDPOINTS_2_0:
            raise ValueError('Invalid region: %s' % region)
        if not secure:
            raise ValueError('CloudSigma driver only supports a secure connection')
        self._host_argument_set = host is not None
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, **kwargs)

    def list_nodes(self, ex_tag=None):
        """
        List available nodes.

        :param ex_tag: If specified, only return servers tagged with the
                       provided tag.
        :type ex_tag: :class:`CloudSigmaTag`
        """
        if ex_tag:
            action = '/tags/%s/servers/detail/' % ex_tag.id
        else:
            action = '/servers/detail/'
        response = self.connection.request(action=action, method='GET').object
        nodes = [self._to_node(data=item) for item in response['objects']]
        return nodes

    def list_sizes(self):
        """
        List available sizes.
        """
        sizes = []
        for value in INSTANCE_TYPES:
            key = value['id']
            size = CloudSigmaNodeSize(id=value['id'], name=value['name'], cpu=value['cpu'], ram=value['memory'], disk=value['disk'], bandwidth=value['bandwidth'], price=self._get_size_price(size_id=key), driver=self.connection.driver)
            sizes.append(size)
        return sizes

    def list_images(self):
        """
        Return a list of available pre-installed library drives.

        Note: If you want to list all the available library drives (both
        pre-installed and installation CDs), use :meth:`ex_list_library_drives`
        method.
        """
        response = self.connection.request(action='/libdrives/').object
        images = [self._to_image(data=item) for item in response['objects']]
        images = [image for image in images if image.extra['image_type'] == 'preinst']
        return images

    def create_node(self, name, size, image, ex_metadata=None, ex_vnc_password=None, ex_avoid=None, ex_vlan=None, public_keys=None):
        """
        Create a new server.

        Server creation consists multiple steps depending on the type of the
        image used.

        1. Installation CD:

            1. Create a server and attach installation cd
            2. Start a server

        2. Pre-installed image:

            1. Clone provided library drive so we can use it
            2. Resize cloned drive to the desired size
            3. Create a server and attach cloned drive
            4. Start a server

        :param ex_metadata: Key / value pairs to associate with the
                            created node. (optional)
        :type ex_metadata: ``dict``

        :param ex_vnc_password: Password to use for VNC access. If not
                                provided, random password is generated.
        :type ex_vnc_password: ``str``

        :param ex_avoid: A list of server UUIDs to avoid when starting this
                         node. (optional)
        :type ex_avoid: ``list``

        :param ex_vlan: Optional UUID of a VLAN network to use. If specified,
                        server will have two nics assigned - 1 with a public ip
                        and 1 with the provided VLAN.
        :type ex_vlan: ``str``

        :param public_keys: Optional list of SSH key UUIDs
        :type public_keys: ``list`` of ``str``
        """
        is_installation_cd = self._is_installation_cd(image=image)
        if ex_vnc_password:
            vnc_password = ex_vnc_password
        else:
            vnc_password = get_secure_random_string(size=12)
        drive_name = '%s-drive' % name
        drive_size = size.disk
        if not is_installation_cd:
            drive = self.ex_clone_drive(drive=image, name=drive_name)
            drive = self._wait_for_drive_state_transition(drive=drive, state='unmounted')
            if drive_size > drive.size:
                drive = self.ex_resize_drive(drive=drive, size=drive_size)
            drive = self._wait_for_drive_state_transition(drive=drive, state='unmounted')
        else:
            drive = image
        data = {}
        data['name'] = name
        data['cpu'] = size.cpu * 2000
        data['mem'] = size.ram * 1024 * 1024
        data['vnc_password'] = vnc_password
        if public_keys:
            data['pubkeys'] = public_keys
        if ex_metadata:
            data['meta'] = ex_metadata
        nic = {'boot_order': None, 'ip_v4_conf': {'conf': 'dhcp'}, 'ip_v6_conf': None}
        nics = [nic]
        if ex_vlan:
            nic = {'boot_order': None, 'ip_v4_conf': None, 'ip_v6_conf': None, 'vlan': ex_vlan}
            nics.append(nic)
        if is_installation_cd:
            device_type = 'ide'
        else:
            device_type = 'virtio'
        drive = {'boot_order': 1, 'dev_channel': '0:0', 'device': device_type, 'drive': drive.id}
        drives = [drive]
        data['nics'] = nics
        data['drives'] = drives
        action = '/servers/'
        response = self.connection.request(action=action, method='POST', data=data)
        node = self._to_node(response.object['objects'][0])
        self.ex_start_node(node=node, ex_avoid=ex_avoid)
        return node

    def destroy_node(self, node, ex_delete_drives=False):
        """
        Destroy the node and all the associated drives.

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        action = '/servers/%s/' % node.id
        if ex_delete_drives is True:
            params = {'recurse': 'all_drives'}
        else:
            params = None
        response = self.connection.request(action=action, method='DELETE', params=params)
        return response.status == httplib.NO_CONTENT

    def reboot_node(self, node):
        """
        Reboot a node.

        Because Cloudsigma API does not provide native reboot call,
        it's emulated using stop and start.

        :param node: Node to reboot.
        :type node: :class:`libcloud.compute.base.Node`
        """
        state = node.state
        if state == NodeState.RUNNING:
            stopped = self.stop_node(node)
        else:
            stopped = True
        if not stopped:
            raise CloudSigmaException('Could not stop node with id %s' % node.id)
        success = False
        for _ in range(5):
            try:
                success = self.start_node(node)
            except CloudSigmaError:
                time.sleep(1)
                continue
            else:
                break
        return success

    def ex_edit_node(self, node, params):
        """
        Edit a node.

        :param node: Node to edit.
        :type node: :class:`libcloud.compute.base.Node`

        :param params: Node parameters to update.
        :type params: ``dict``

        :return Edited node.
        :rtype: :class:`libcloud.compute.base.Node`
        """
        data = {}
        data['name'] = node.name
        data['cpu'] = node.extra['cpus']
        data['mem'] = node.extra['memory']
        data['vnc_password'] = node.extra['vnc_password']
        nics = copy.deepcopy(node.extra.get('nics', []))
        data['nics'] = nics
        data.update(params)
        action = '/servers/%s/' % node.id
        response = self.connection.request(action=action, method='PUT', data=data).object
        node = self._to_node(data=response)
        return node

    def start_node(self, node, ex_avoid=None):
        """
        Start a node.

        :param node: Node to start.
        :type node: :class:`libcloud.compute.base.Node`

        :param ex_avoid: A list of other server uuids to avoid when
                         starting this node. If provided, node will
                         attempt to be started on a different
                         physical infrastructure from other servers
                         specified using this argument. (optional)
        :type ex_avoid: ``list``
        """
        params = {}
        if ex_avoid:
            params['avoid'] = ','.join(ex_avoid)
        path = '/servers/%s/action/' % node.id
        response = self._perform_action(path=path, action='start', params=params, method='POST')
        return response.status == httplib.ACCEPTED

    def stop_node(self, node):
        path = '/servers/%s/action/' % node.id
        response = self._perform_action(path=path, action='stop', method='POST')
        return response.status == httplib.ACCEPTED

    def ex_start_node(self, node, ex_avoid=None):
        return self.start_node(node=node, ex_avoid=ex_avoid)

    def ex_stop_node(self, node):
        """
        Stop a node.
        """
        return self.stop_node(node=node)

    def ex_clone_node(self, node, name=None, random_vnc_password=None):
        """
        Clone the provided node.

        :param name: Optional name for the cloned node.
        :type name: ``str``
        :param random_vnc_password: If True, a new random VNC password will be
                                    generated for the cloned node. Otherwise
                                    password from the cloned node will be
                                    reused.
        :type random_vnc_password: ``bool``

        :return: Cloned node.
        :rtype: :class:`libcloud.compute.base.Node`
        """
        data = {}
        data['name'] = name
        data['random_vnc_password'] = random_vnc_password
        path = '/servers/%s/action/' % node.id
        response = self._perform_action(path=path, action='clone', method='POST', data=data).object
        node = self._to_node(data=response)
        return node

    def ex_get_node(self, node_id, return_json=False):
        action = '/servers/%s/' % node_id
        response = self.connection.request(action=action).object
        if return_json is True:
            return response
        return self._to_node(response)

    def ex_open_vnc_tunnel(self, node):
        """
        Open a VNC tunnel to the provided node and return the VNC url.

        :param node: Node to open the VNC tunnel to.
        :type node: :class:`libcloud.compute.base.Node`

        :return: URL of the opened VNC tunnel.
        :rtype: ``str``
        """
        path = '/servers/%s/action/' % node.id
        response = self._perform_action(path=path, action='open_vnc', method='POST').object
        vnc_url = response['vnc_url']
        return vnc_url

    def ex_close_vnc_tunnel(self, node):
        """
        Close a VNC server to the provided node.

        :param node: Node to close the VNC tunnel to.
        :type node: :class:`libcloud.compute.base.Node`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        path = '/servers/%s/action/' % node.id
        response = self._perform_action(path=path, action='close_vnc', method='POST')
        return response.status == httplib.ACCEPTED

    def ex_list_library_drives(self):
        """
        Return a list of all the available library drives (pre-installed and
        installation CDs).

        :rtype: ``list`` of :class:`.CloudSigmaDrive` objects
        """
        response = self.connection.request(action='/libdrives/').object
        drives = [self._to_drive(data=item) for item in response['objects']]
        return drives

    def ex_list_user_drives(self):
        """
        Return a list of all the available user's drives.

        :rtype: ``list`` of :class:`.CloudSigmaDrive` objects
        """
        response = self.connection.request(action='/drives/detail/').object
        drives = [self._to_drive(data=item) for item in response['objects']]
        return drives

    def list_volumes(self):
        return self.ex_list_user_drives()

    def ex_create_drive(self, name, size, media='disk', ex_avoid=None):
        """
        Create a new drive.

        :param name: Drive name.
        :type name: ``str``

        :param size: Drive size in GBs.
        :type size: ``int``

        :param media: Drive media type (cdrom, disk).
        :type media: ``str``

        :param ex_avoid: A list of other drive uuids to avoid when
                         creating this drive. If provided, drive will
                         attempt to be created on a different
                         physical infrastructure from other drives
                         specified using this argument. (optional)
        :type ex_avoid: ``list``

        :return: Created drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
        params = {}
        data = {'name': name, 'size': size * 1024 * 1024 * 1024, 'media': media}
        if ex_avoid:
            params['avoid'] = ','.join(ex_avoid)
        action = '/drives/'
        response = self.connection.request(action=action, method='POST', params=params, data=data).object
        drive = self._to_drive(data=response['objects'][0])
        return drive

    def create_volume(self, name, size, media='disk', ex_avoid=None):
        return self.ex_create_drive(name=name, size=size, media=media, ex_avoid=ex_avoid)

    def ex_clone_drive(self, drive, name=None, ex_avoid=None):
        """
        Clone a library or a standard drive.

        :param drive: Drive to clone.
        :type drive: :class:`libcloud.compute.base.NodeImage` or
                     :class:`.CloudSigmaDrive`

        :param name: Optional name for the cloned drive.
        :type name: ``str``

        :param ex_avoid: A list of other drive uuids to avoid when
                         creating this drive. If provided, drive will
                         attempt to be created on a different
                         physical infrastructure from other drives
                         specified using this argument. (optional)
        :type ex_avoid: ``list``

        :return: New cloned drive.
        :rtype: :class:`.CloudSigmaDrive`
        """
        params = {}
        data = {}
        if ex_avoid:
            params['avoid'] = ','.join(ex_avoid)
        if name:
            data['name'] = name
        path = '/drives/%s/action/' % drive.id
        response = self._perform_action(path=path, action='clone', params=params, data=data, method='POST')
        drive = self._to_drive(data=response.object['objects'][0])
        return drive

    def ex_resize_drive(self, drive, size):
        """
        Resize a drive.

        :param drive: Drive to resize.

        :param size: New drive size in GBs.
        :type size: ``int``

        :return: Drive object which is being resized.
        :rtype: :class:`.CloudSigmaDrive`
        """
        path = '/drives/%s/action/' % drive.id
        data = {'name': drive.name, 'size': size * 1024 * 1024 * 1024, 'media': 'disk'}
        response = self._perform_action(path=path, action='resize', method='POST', data=data)
        drive = self._to_drive(data=response.object['objects'][0])
        return drive

    def ex_attach_drive(self, node, drive):
        """
        Attach drive to node

        :param node: Node to attach the drive to.
        :type node: :class:`libcloud.compute.base.Node`

        :param drive: Drive to attach.
        :type drive: :class:`.CloudSigmaDrive`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        data = self.ex_get_node(node.id, return_json=True)
        dev_channels = [item['dev_channel'] for item in data['drives']]
        dev_channel = None
        for controller in range(MAX_VIRTIO_CONTROLLERS):
            for unit in range(MAX_VIRTIO_UNITS):
                if '{}:{}'.format(controller, unit) not in dev_channels:
                    dev_channel = '{}:{}'.format(controller, unit)
                    break
            if dev_channel:
                break
        else:
            raise CloudSigmaException('Could not attach drive to %s' % node.id)
        item = {'boot_order': None, 'dev_channel': dev_channel, 'device': 'virtio', 'drive': str(drive.id)}
        data['drives'].append(item)
        action = '/servers/%s/' % node.id
        response = self.connection.request(action=action, data=data, method='PUT')
        return response.status == 200

    def attach_volume(self, node, volume):
        return self.ex_attach_drive(node=node, drive=volume)

    def ex_detach_drive(self, node, drive):
        data = self.ex_get_node(node.id, return_json=True)
        data['drives'] = [item for item in data['drives'] if item['drive']['uuid'] != drive.id]
        action = '/servers/%s/' % node.id
        response = self.connection.request(action=action, data=data, method='PUT')
        return response.status == 200

    def detach_volume(self, node, volume):
        return self.ex_detach_drive(node=node, drive=volume)

    def ex_get_drive(self, drive_id):
        """
        Retrieve information about a single drive.

        :param drive_id: ID of the drive to retrieve.
        :type drive_id: ``str``

        :return: Drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
        action = '/drives/%s/' % drive_id
        response = self.connection.request(action=action).object
        drive = self._to_drive(data=response)
        return drive

    def ex_destroy_drive(self, drive):
        action = '/drives/%s/' % drive.id
        response = self.connection.request(action=action, method='DELETE')
        return response.status == httplib.NO_CONTENT

    def destroy_volume(self, drive):
        return self.ex_destroy_drive(drive=drive)

    def ex_list_firewall_policies(self):
        """
        List firewall policies.

        :rtype: ``list`` of :class:`.CloudSigmaFirewallPolicy`
        """
        action = '/fwpolicies/detail/'
        response = self.connection.request(action=action, method='GET').object
        policies = [self._to_firewall_policy(data=item) for item in response['objects']]
        return policies

    def ex_create_firewall_policy(self, name, rules=None):
        """
        Create a firewall policy.

        :param name: Policy name.
        :type name: ``str``

        :param rules: List of firewall policy rules to associate with this
                      policy. (optional)
        :type rules: ``list`` of ``dict``

        :return: Created firewall policy object.
        :rtype: :class:`.CloudSigmaFirewallPolicy`
        """
        data = {}
        obj = {}
        obj['name'] = name
        if rules:
            obj['rules'] = rules
        data['objects'] = [obj]
        action = '/fwpolicies/'
        response = self.connection.request(action=action, method='POST', data=data).object
        policy = self._to_firewall_policy(data=response['objects'][0])
        return policy

    def ex_attach_firewall_policy(self, policy, node, nic_mac=None):
        """
        Attach firewall policy to a public NIC interface on the server.

        :param policy: Firewall policy to attach.
        :type policy: :class:`.CloudSigmaFirewallPolicy`

        :param node: Node to attach policy to.
        :type node: :class:`libcloud.compute.base.Node`

        :param nic_mac: Optional MAC address of the NIC to add the policy to.
                        If not specified, first public interface is used
                        instead.
        :type nic_mac: ``str``

        :return: Node object to which the policy was attached to.
        :rtype: :class:`libcloud.compute.base.Node`
        """
        nics = copy.deepcopy(node.extra.get('nics', []))
        if nic_mac:
            nic = [n for n in nics if n['mac'] == nic_mac]
        else:
            nic = nics
        if len(nic) == 0:
            raise ValueError('Cannot find the NIC interface to attach a policy to')
        nic = nic[0]
        nic['firewall_policy'] = policy.id
        params = {'nics': nics}
        node = self.ex_edit_node(node=node, params=params)
        return node

    def ex_delete_firewall_policy(self, policy):
        """
        Delete a firewall policy.

        :param policy: Policy to delete to.
        :type policy: :class:`.CloudSigmaFirewallPolicy`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        action = '/fwpolicies/%s/' % policy.id
        response = self.connection.request(action=action, method='DELETE')
        return response.status == httplib.NO_CONTENT

    def ex_list_servers_availability_groups(self):
        """
        Return which running servers share the same physical compute host.

        :return: A list of server UUIDs which share the same physical compute
                 host. Servers which share the same host will be stored under
                 the same list index.
        :rtype: ``list`` of ``list``
        """
        action = '/servers/availability_groups/'
        response = self.connection.request(action=action, method='GET')
        return response.object

    def ex_list_drives_availability_groups(self):
        """
        Return which drives share the same physical storage host.

        :return: A list of drive UUIDs which share the same physical storage
                 host. Drives which share the same host will be stored under
                 the same list index.
        :rtype: ``list`` of ``list``
        """
        action = '/drives/availability_groups/'
        response = self.connection.request(action=action, method='GET')
        return response.object

    def ex_list_tags(self):
        """
        List all the available tags.

        :rtype: ``list`` of :class:`.CloudSigmaTag` objects
        """
        action = '/tags/detail/'
        response = self.connection.request(action=action, method='GET').object
        tags = [self._to_tag(data=item) for item in response['objects']]
        return tags

    def ex_get_tag(self, tag_id):
        """
        Retrieve a single tag.

        :param tag_id: ID of the tag to retrieve.
        :type tag_id: ``str``

        :rtype: ``list`` of :class:`.CloudSigmaTag` objects
        """
        action = '/tags/%s/' % tag_id
        response = self.connection.request(action=action, method='GET').object
        tag = self._to_tag(data=response)
        return tag

    def ex_create_tag(self, name, resource_uuids=None):
        """
        Create a tag.

        :param name: Tag name.
        :type name: ``str``

        :param resource_uuids: Optional list of resource UUIDs to assign this
                               tag go.
        :type resource_uuids: ``list`` of ``str``

        :return: Created tag object.
        :rtype: :class:`.CloudSigmaTag`
        """
        data = {}
        data['objects'] = [{'name': name}]
        if resource_uuids:
            data['resources'] = resource_uuids
        action = '/tags/'
        response = self.connection.request(action=action, method='POST', data=data).object
        tag = self._to_tag(data=response['objects'][0])
        return tag

    def ex_tag_resource(self, resource, tag):
        """
        Associate tag with the provided resource.

        :param resource: Resource to associate a tag with.
        :type resource: :class:`libcloud.compute.base.Node` or
                        :class:`.CloudSigmaDrive`

        :param tag: Tag to associate with the resources.
        :type tag: :class:`.CloudSigmaTag`

        :return: Updated tag object.
        :rtype: :class:`.CloudSigmaTag`
        """
        if not hasattr(resource, 'id'):
            raise ValueError("Resource doesn't have id attribute")
        return self.ex_tag_resources(resources=[resource], tag=tag)

    def ex_tag_resources(self, resources, tag):
        """
        Associate tag with the provided resources.

        :param resources: Resources to associate a tag with.
        :type resources: ``list`` of :class:`libcloud.compute.base.Node` or
                        :class:`.CloudSigmaDrive`

        :param tag: Tag to associate with the resources.
        :type tag: :class:`.CloudSigmaTag`

        :return: Updated tag object.
        :rtype: :class:`.CloudSigmaTag`
        """
        resources = tag.resources[:]
        for resource in resources:
            if not hasattr(resource, 'id'):
                raise ValueError("Resource doesn't have id attribute")
            resources.append(resource.id)
        resources = list(set(resources))
        data = {'name': tag.name, 'resources': resources}
        action = '/tags/%s/' % tag.id
        response = self.connection.request(action=action, method='PUT', data=data).object
        tag = self._to_tag(data=response)
        return tag

    def ex_delete_tag(self, tag):
        """
        Delete a tag.

        :param tag: Tag to delete.
        :type tag: :class:`.CloudSigmaTag`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        action = '/tags/%s/' % tag.id
        response = self.connection.request(action=action, method='DELETE')
        return response.status == httplib.NO_CONTENT

    def ex_get_balance(self):
        """
        Retrieve account balance information.

        :return: Dictionary with two items ("balance" and "currency").
        :rtype: ``dict``
        """
        action = '/balance/'
        response = self.connection.request(action=action, method='GET')
        return response.object

    def ex_get_pricing(self):
        """
        Retrieve pricing information that are applicable to the cloud.

        :return: Dictionary with pricing information.
        :rtype: ``dict``
        """
        action = '/pricing/'
        response = self.connection.request(action=action, method='GET')
        return response.object

    def ex_get_usage(self):
        """
        Retrieve account current usage information.

        :return: Dictionary with two items ("balance" and "usage").
        :rtype: ``dict``
        """
        action = '/currentusage/'
        response = self.connection.request(action=action, method='GET')
        return response.object

    def ex_list_subscriptions(self, status='all', resources=None):
        """
        List subscriptions for this account.

        :param status: Only return subscriptions with the provided status
                       (optional).
        :type status: ``str``
        :param resources: Only return subscriptions for the provided resources
                          (optional).
        :type resources: ``list``

        :rtype: ``list``
        """
        params = {}
        if status:
            params['status'] = status
        if resources:
            params['resource'] = ','.join(resources)
        response = self.connection.request(action='/subscriptions/', params=params).object
        subscriptions = self._to_subscriptions(data=response)
        return subscriptions

    def ex_toggle_subscription_auto_renew(self, subscription):
        """
        Toggle subscription auto renew status.

        :param subscription: Subscription to toggle the auto renew flag for.
        :type subscription: :class:`.CloudSigmaSubscription`

        :return: ``True`` on success, ``False`` otherwise.
        :rtype: ``bool``
        """
        path = '/subscriptions/%s/action/' % subscription.id
        response = self._perform_action(path=path, action='auto_renew', method='POST')
        return response.status == httplib.OK

    def ex_create_subscription(self, amount, period, resource, auto_renew=False):
        """
        Create a new subscription.

        :param amount: Subscription amount. For example, in dssd case this
                       would be disk size in gigabytes.
        :type amount: ``int``

        :param period: Subscription period. For example: 30 days, 1 week, 1
                                            month, ...
        :type period: ``str``

        :param resource: Resource the purchase the subscription for.
        :type resource: ``str``

        :param auto_renew: True to automatically renew the subscription.
        :type auto_renew: ``bool``
        """
        data = [{'amount': amount, 'period': period, 'auto_renew': auto_renew, 'resource': resource}]
        response = self.connection.request(action='/subscriptions/', data=data, method='POST')
        data = response.object['objects'][0]
        subscription = self._to_subscription(data=data)
        return subscription

    def ex_list_capabilities(self):
        """
        Retrieve all the basic and sensible limits of the API.

        :rtype: ``dict``
        """
        action = '/capabilities/'
        response = self.connection.request(action=action, method='GET')
        capabilities = response.object
        return capabilities

    def list_key_pairs(self):
        """
        List all the available key pair objects.

        :rtype: ``list`` of :class:`KeyPair` objects
        """
        action = '/keypairs'
        response = self.connection.request(action=action, method='GET').object
        keys = [self._to_key_pair(data=item) for item in response['objects']]
        return keys

    def get_key_pair(self, key_uuid):
        """
        Retrieve a single key pair.

        :param name: The uuid of the key pair to retrieve.
        :type name: ``str``

        :rtype: :class:`.KeyPair`
        """
        action = '/keypairs/%s/' % key_uuid
        response = self.connection.request(action=action, method='GET').object
        return self._to_key_pair(response)

    def create_key_pair(self, name):
        """
        Create a new SSH key.

        :param name: Key pair name.
        :type name: ``str``
        """
        action = '/keypairs/'
        data = {'objects': [{'name': name}]}
        response = self.connection.request(action=action, method='POST', data=data).object
        return self._to_key_pair(response['objects'][0])

    def import_key_pair_from_string(self, name, key_material):
        """
        Import a new public key from string.

        :param name: Key pair name.
        :type name: ``str``

        :param key_material: Public key material.
        :type key_material: ``str``

        :rtype: :class:`.KeyPair` object
        """
        action = '/keypairs/'
        data = {'objects': [{'name': name, 'public_key': key_material.replace('\n', '')}]}
        response = self.connection.request(action=action, method='POST', data=data).object
        return self._to_key_pair(response['objects'][0])

    def delete_key_pair(self, key_pair):
        """
        Delete an existing key pair.

        :param key_pair: Key pair object
        :type key_pair: :class:`.KeyPair`

        :rtype: ``bool``
        """
        action = '/keypairs/%s/' % key_pair.extra['uuid']
        response = self.connection.request(action=action, method='DELETE')
        return response.status == 204

    def _parse_ips_from_nic(self, nic):
        """
        Parse private and public IP addresses from the provided network
        interface object.

        :param nic: NIC object.
        :type nic: ``dict``

        :return: (public_ips, private_ips) tuple.
        :rtype: ``tuple``
        """
        public_ips, private_ips = ([], [])
        ipv4_conf = nic['ip_v4_conf']
        ipv6_conf = nic['ip_v6_conf']
        ip_v4 = ipv4_conf['ip'] if ipv4_conf else None
        ip_v6 = ipv6_conf['ip'] if ipv6_conf else None
        ipv4 = ip_v4['uuid'] if ip_v4 else None
        ipv6 = ip_v4['uuid'] if ip_v6 else None
        ips = []
        if ipv4:
            ips.append(ipv4)
        if ipv6:
            ips.append(ipv6)
        runtime = nic['runtime']
        ip_v4 = runtime['ip_v4'] if nic['runtime'] else None
        ip_v6 = runtime['ip_v6'] if nic['runtime'] else None
        ipv4 = ip_v4['uuid'] if ip_v4 else None
        ipv6 = ip_v4['uuid'] if ip_v6 else None
        if ipv4:
            ips.append(ipv4)
        if ipv6:
            ips.append(ipv6)
        ips = set(ips)
        for ip in ips:
            if is_private_subnet(ip):
                private_ips.append(ip)
            else:
                public_ips.append(ip)
        return (public_ips, private_ips)

    def _to_node(self, data):
        id = data['uuid']
        name = data['name']
        state = self.NODE_STATE_MAP.get(data['status'], NodeState.UNKNOWN)
        public_ips = []
        private_ips = []
        extra = {'cpus': data['cpu'] / 2000, 'memory': data['mem'] / 1024 / 1024, 'nics': data['nics'], 'vnc_password': data['vnc_password'], 'meta': data['meta'], 'runtime': data['runtime'], 'drives': data['drives']}
        image = None
        drive_size = 0
        for item in extra['drives']:
            if item['boot_order'] == 1:
                drive = self.ex_get_drive(item['drive']['uuid'])
                drive_size = drive.size
                image = '{} {}'.format(drive.extra.get('distribution', ''), drive.extra.get('version', ''))
                break
        try:
            kwargs = SPECS_TO_SIZE[extra['cpus'], extra['memory'], drive_size]
            size = CloudSigmaNodeSize(**kwargs, driver=self)
        except KeyError:
            id_to_hash = str(extra['cpus']) + str(extra['memory']) + str(drive_size)
            size_id = hashlib.md5(id_to_hash.encode('utf-8')).hexdigest()
            size_name = 'custom, {} CPUs, {}MB RAM, {}GB disk'.format(extra['cpus'], extra['memory'], drive_size)
            size = CloudSigmaNodeSize(id=size_id, name=size_name, cpu=extra['cpus'], ram=extra['memory'], disk=drive_size, bandwidth=None, price=0, driver=self)
        for nic in data['nics']:
            _public_ips, _private_ips = self._parse_ips_from_nic(nic=nic)
            public_ips.extend(_public_ips)
            private_ips.extend(_private_ips)
        node = Node(id=id, name=name, state=state, public_ips=public_ips, image=image, private_ips=private_ips, driver=self, size=size, extra=extra)
        return node

    def _to_image(self, data):
        extra_keys = ['description', 'arch', 'image_type', 'os', 'licenses', 'media', 'meta']
        id = data['uuid']
        name = data['name']
        extra = self._extract_values(obj=data, keys=extra_keys)
        image = NodeImage(id=id, name=name, driver=self, extra=extra)
        return image

    def _to_drive(self, data):
        id = data['uuid']
        name = data['name']
        size = data['size'] / 1024 / 1024 / 1024
        media = data['media']
        status = data['status']
        extra = {'mounted_on': data.get('mounted_on', []), 'storage_type': data.get('storage_type', ''), 'distribution': data['meta'].get('distribution', ''), 'version': data['meta'].get('version', ''), 'os': data['meta'].get('os', ''), 'paid': data['meta'].get('paid', ''), 'architecture': data['meta'].get('arch', ''), 'created_at': data['meta'].get('created_at', '')}
        drive = CloudSigmaDrive(id=id, name=name, size=size, media=media, status=status, driver=self, extra=extra)
        return drive

    def _to_tag(self, data):
        resources = data['resources']
        resources = [resource['uuid'] for resource in resources]
        tag = CloudSigmaTag(id=data['uuid'], name=data['name'], resources=resources)
        return tag

    def _to_subscriptions(self, data):
        subscriptions = []
        for item in data['objects']:
            subscription = self._to_subscription(data=item)
            subscriptions.append(subscription)
        return subscriptions

    def _to_subscription(self, data):
        if data.get('start_time', None):
            start_time = parse_date(data['start_time'])
        else:
            start_time = None
        if data.get('end_time', None):
            end_time = parse_date(data['end_time'])
        else:
            end_time = None
        obj_uuid = data['subscribed_object']
        subscription = CloudSigmaSubscription(id=data['id'], resource=data['resource'], amount=int(data['amount']), period=data['period'], status=data['status'], price=data['price'], start_time=start_time, end_time=end_time, auto_renew=data['auto_renew'], subscribed_object=obj_uuid)
        return subscription

    def _to_firewall_policy(self, data):
        rules = []
        for item in data.get('rules', []):
            rule = CloudSigmaFirewallPolicyRule(action=item['action'], direction=item['direction'], ip_proto=item['ip_proto'], src_ip=item['src_ip'], src_port=item['src_port'], dst_ip=item['dst_ip'], dst_port=item['dst_port'], comment=item['comment'])
            rules.append(rule)
        policy = CloudSigmaFirewallPolicy(id=data['uuid'], name=data['name'], rules=rules)
        return policy

    def _to_key_pair(self, data):
        extra = {'uuid': data['uuid'], 'tags': data['tags'], 'resource_uri': data['resource_uri'], 'permissions': data['permissions'], 'meta': data['meta']}
        return KeyPair(name=data['name'], public_key=data['public_key'], fingerprint=data['fingerprint'], driver=self, private_key=data['private_key'], extra=extra)

    def _perform_action(self, path, action, method='POST', params=None, data=None):
        """
        Perform API action and return response object.
        """
        if params:
            params = params.copy()
        else:
            params = {}
        params['do'] = action
        response = self.connection.request(action=path, method=method, params=params, data=data)
        return response

    def _is_installation_cd(self, image):
        """
        Detect if the provided image is an installation CD.

        :rtype: ``bool``
        """
        if isinstance(image, CloudSigmaDrive) and image.media == 'cdrom':
            return True
        return False

    def _extract_values(self, obj, keys):
        """
        Extract values from a dictionary and return a new dictionary with
        extracted values.

        :param obj: Dictionary to extract values from.
        :type obj: ``dict``

        :param keys: Keys to extract.
        :type keys: ``list``

        :return: Dictionary with extracted values.
        :rtype: ``dict``
        """
        result = {}
        for key in keys:
            result[key] = obj[key]
        return result

    def _wait_for_drive_state_transition(self, drive, state, timeout=DRIVE_TRANSITION_TIMEOUT):
        """
        Wait for a drive to transition to the provided state.

        Note: This function blocks and periodically calls "GET drive" endpoint
        to check if the drive has already transitioned to the desired state.

        :param drive: Drive to wait for.
        :type drive: :class:`.CloudSigmaDrive`

        :param state: Desired drive state.
        :type state: ``str``

        :param timeout: How long to wait for the transition (in seconds) before
                        timing out.
        :type timeout: ``int``

        :return: Drive object.
        :rtype: :class:`.CloudSigmaDrive`
        """
        start_time = time.time()
        while drive.status != state:
            drive = self.ex_get_drive(drive_id=drive.id)
            if drive.status == state:
                break
            current_time = time.time()
            delta = current_time - start_time
            if delta >= timeout:
                msg = 'Timed out while waiting for drive transition (timeout=%s seconds)' % timeout
                raise Exception(msg)
            time.sleep(self.DRIVE_TRANSITION_SLEEP_INTERVAL)
        return drive

    def _ex_connection_class_kwargs(self):
        """
        Return the host value based on the user supplied region.
        """
        kwargs = {}
        if not self._host_argument_set:
            kwargs['host'] = API_ENDPOINTS_2_0[self.region]['host']
        return kwargs