import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
class VultrNodeDriverV1(VultrNodeDriver):
    """
    VultrNode node driver.
    """
    connectionCls = VultrConnection
    NODE_STATE_MAP = {'pending': NodeState.PENDING, 'active': NodeState.RUNNING}
    EX_CREATE_YES_NO_ATTRIBUTES = ['enable_ipv6', 'enable_private_network', 'auto_backups', 'notify_activate', 'ddos_protection']
    EX_CREATE_ID_ATTRIBUTES = {'iso_id': 'ISOID', 'script_id': 'SCRIPTID', 'snapshot_id': 'SNAPSHOTID', 'app_id': 'APPID'}
    EX_CREATE_ATTRIBUTES = ['ipxe_chain_url', 'label', 'userdata', 'reserved_ip_v4', 'hostname', 'tag']
    EX_CREATE_ATTRIBUTES.extend(EX_CREATE_YES_NO_ATTRIBUTES)
    EX_CREATE_ATTRIBUTES.extend(EX_CREATE_ID_ATTRIBUTES.keys())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._helper = VultrNodeDriverHelper()

    def list_nodes(self):
        return self._list_resources('/v1/server/list', self._to_node)

    def list_key_pairs(self):
        """
        List all the available SSH keys.
        :return: Available SSH keys.
        :rtype: ``list`` of :class:`SSHKey`
        """
        return self._list_resources('/v1/sshkey/list', self._to_ssh_key)

    def create_key_pair(self, name, public_key=''):
        """
        Create a new SSH key.
        :param name: Name of the new SSH key
        :type name: ``str``

        :key public_key: Public part of the new SSH key
        :type name: ``str``

        :return: True on success
        :rtype: ``bool``
        """
        params = {'name': name, 'ssh_key': public_key}
        res = self.connection.post('/v1/sshkey/create', params)
        return res.status == httplib.OK

    def delete_key_pair(self, key_pair):
        """
        Delete an SSH key.
        :param key_pair: The SSH key to delete
        :type key_pair: :class:`SSHKey`

        :return: True on success
        :rtype: ``bool``
        """
        params = {'SSHKEYID': key_pair.id}
        res = self.connection.post('/v1/sshkey/destroy', params)
        return res.status == httplib.OK

    def list_locations(self):
        return self._list_resources('/v1/regions/list', self._to_location)

    def list_sizes(self):
        return self._list_resources('/v1/plans/list', self._to_size)

    def list_images(self):
        return self._list_resources('/v1/os/list', self._to_image)

    def create_node(self, name, size, image, location, ex_ssh_key_ids=None, ex_create_attr=None):
        """
        Create a node

        :param name: Name for the new node
        :type name: ``str``

        :param size: Size of the new node
        :type size: :class:`NodeSize`

        :param image: Image for the new node
        :type image: :class:`NodeImage`

        :param location: Location of the new node
        :type location: :class:`NodeLocation`

        :param ex_ssh_key_ids: IDs of the SSH keys to initialize
        :type ex_sshkeyid: ``list`` of ``str``

        :param ex_create_attr: Extra attributes for node creation
        :type ex_create_attr: ``dict``

        The `ex_create_attr` parameter can include the following dictionary
        key and value pairs:

        * `ipxe_chain_url`: ``str`` for specifying URL to boot via IPXE
        * `iso_id`: ``str`` the ID of a specific ISO to mount,
          only meaningful with the `Custom` `NodeImage`
        * `script_id`: ``int`` ID of a startup script to execute on boot,
          only meaningful when the `NodeImage` is not `Custom`
        * 'snapshot_id`: ``str`` Snapshot ID to restore for the initial
          installation, only meaningful with the `Snapshot` `NodeImage`
        * `enable_ipv6`: ``bool`` Whether an IPv6 subnet should be assigned
        * `enable_private_network`: ``bool`` Whether private networking
          support should be added
        * `label`: ``str`` Text label to be shown in the control panel
        * `auto_backups`: ``bool`` Whether automatic backups should be enabled
        * `app_id`: ``int`` App ID to launch if launching an application,
          only meaningful when the `NodeImage` is `Application`
        * `userdata`: ``str`` Base64 encoded cloud-init user-data
        * `notify_activate`: ``bool`` Whether an activation email should be
          sent when the server is ready
        * `ddos_protection`: ``bool`` Whether DDOS protection should be enabled
        * `reserved_ip_v4`: ``str`` IP address of the floating IP to use as
          the main IP of this server
        * `hostname`: ``str`` The hostname to assign to this server
        * `tag`: ``str`` The tag to assign to this server

        :return: The newly created node.
        :rtype: :class:`Node`

        """
        params = {'DCID': location.id, 'VPSPLANID': size.id, 'OSID': image.id, 'label': name}
        if ex_ssh_key_ids is not None:
            params['SSHKEYID'] = ','.join(ex_ssh_key_ids)
        ex_create_attr = ex_create_attr or {}
        for key, value in ex_create_attr.items():
            if key in self.EX_CREATE_ATTRIBUTES:
                if key in self.EX_CREATE_YES_NO_ATTRIBUTES:
                    params[key] = 'yes' if value else 'no'
                else:
                    if key in self.EX_CREATE_ID_ATTRIBUTES:
                        key = self.EX_CREATE_ID_ATTRIBUTES[key]
                    params[key] = value
        result = self.connection.post('/v1/server/create', params)
        if result.status != httplib.OK:
            return False
        subid = result.object['SUBID']
        retry_count = 3
        created_node = None
        for _ in range(retry_count):
            try:
                nodes = self.list_nodes()
                created_node = [n for n in nodes if n.id == subid][0]
            except IndexError:
                time.sleep(1)
            else:
                break
        return created_node

    def reboot_node(self, node):
        params = {'SUBID': node.id}
        res = self.connection.post('/v1/server/reboot', params)
        return res.status == httplib.OK

    def destroy_node(self, node):
        params = {'SUBID': node.id}
        res = self.connection.post('/v1/server/destroy', params)
        return res.status == httplib.OK

    def _list_resources(self, url, tranform_func):
        data = self.connection.get(url).object
        sorted_key = sorted(data)
        return [tranform_func(data[key]) for key in sorted_key]

    def _to_node(self, data):
        if 'status' in data:
            state = self.NODE_STATE_MAP.get(data['status'], NodeState.UNKNOWN)
            if state == NodeState.RUNNING and data['power_status'] != 'running':
                state = NodeState.STOPPED
        else:
            state = NodeState.UNKNOWN
        if 'main_ip' in data and data['main_ip'] is not None:
            public_ips = [data['main_ip']]
        else:
            public_ips = []
        if len(data['internal_ip']) > 0:
            private_ips = [data['internal_ip']]
        else:
            private_ips = []
        created_at = parse_date(data['date_created'])
        extra_keys = ['location', 'default_password', 'pending_charges', 'cost_per_month', 'current_bandwidth_gb', 'allowed_bandwidth_gb', 'netmask_v4', 'gateway_v4', 'power_status', 'server_state', 'v6_networks', 'kvm_url', 'auto_backups', 'tag', 'APPID', 'FIREWALLGROUPID']
        extra = self._helper.handle_extra(extra_keys, data)
        resolve_data = VULTR_COMPUTE_INSTANCE_IMAGES.get(data['OSID'])
        if resolve_data:
            image = self._to_image(resolve_data)
        else:
            image = None
        resolve_data = VULTR_COMPUTE_INSTANCE_SIZES.get(data['VPSPLANID'])
        if resolve_data:
            size = self._to_size(resolve_data)
        else:
            size = None
        node = Node(id=data['SUBID'], name=data['label'], state=state, public_ips=public_ips, private_ips=private_ips, image=image, size=size, extra=extra, created_at=created_at, driver=self)
        return node

    def _to_location(self, data):
        extra_keys = ['continent', 'state', 'ddos_protection', 'block_storage', 'regioncode']
        extra = self._helper.handle_extra(extra_keys, data)
        return NodeLocation(id=data['DCID'], name=data['name'], country=data['country'], extra=extra, driver=self)

    def _to_size(self, data):
        extra_keys = ['vcpu_count', 'plan_type', 'available_locations']
        extra = self._helper.handle_extra(extra_keys, data)
        if extra.get('vcpu_count').isdigit():
            extra['vcpu_count'] = int(extra['vcpu_count'])
        ram = int(data['ram'])
        disk = int(data['disk'])
        bandwidth = int(float(data['bandwidth']))
        price = float(data['price_per_month'])
        return NodeSize(id=data['VPSPLANID'], name=data['name'], ram=ram, disk=disk, bandwidth=bandwidth, price=price, extra=extra, driver=self)

    def _to_image(self, data):
        extra_keys = ['arch', 'family']
        extra = self._helper.handle_extra(extra_keys, data)
        return NodeImage(id=data['OSID'], name=data['name'], extra=extra, driver=self)

    def _to_ssh_key(self, data):
        return SSHKey(id=data['SSHKEYID'], name=data['name'], pub_key=data['ssh_key'])