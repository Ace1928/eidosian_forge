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
class CloudSigma_1_0_NodeDriver(CloudSigmaNodeDriver):
    type = Provider.CLOUDSIGMA
    name = 'CloudSigma (API v1.0)'
    website = 'http://www.cloudsigma.com/'
    connectionCls = CloudSigma_1_0_Connection
    IMAGING_TIMEOUT = 20 * 60
    NODE_STATE_MAP = {'active': NodeState.RUNNING, 'stopped': NodeState.TERMINATED, 'dead': NodeState.TERMINATED, 'dumped': NodeState.TERMINATED}

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region=DEFAULT_REGION, **kwargs):
        if region not in API_ENDPOINTS_1_0:
            raise ValueError('Invalid region: %s' % region)
        self._host_argument_set = host is not None
        self.api_name = 'cloudsigma_%s' % region
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, **kwargs)

    def reboot_node(self, node):
        """
        Reboot a node.

        Because Cloudsigma API does not provide native reboot call,
        it's emulated using stop and start.

        @inherits: :class:`NodeDriver.reboot_node`
        """
        node = self._get_node(node.id)
        state = node.state
        if state == NodeState.RUNNING:
            stopped = self.ex_stop_node(node)
        else:
            stopped = True
        if not stopped:
            raise CloudSigmaException('Could not stop node with id %s' % node.id)
        success = self.ex_start_node(node)
        return success

    def destroy_node(self, node):
        """
        Destroy a node (all the drives associated with it are NOT destroyed).

        If a node is still running, it's stopped before it's destroyed.

        @inherits: :class:`NodeDriver.destroy_node`
        """
        node = self._get_node(node.id)
        state = node.state
        if state == NodeState.RUNNING:
            stopped = self.ex_stop_node(node)
        else:
            stopped = True
        if not stopped:
            raise CloudSigmaException('Could not stop node with id %s' % node.id)
        response = self.connection.request(action='/servers/%s/destroy' % node.id, method='POST')
        return response.status == 204

    def list_images(self, location=None):
        """
        Return a list of available standard images (this call might take up
        to 15 seconds to return).

        @inherits: :class:`NodeDriver.list_images`
        """
        response = self.connection.request(action='/drives/standard/info').object
        images = []
        for value in response:
            if value.get('type'):
                if value['type'] == 'disk':
                    image = NodeImage(id=value['drive'], name=value['name'], driver=self.connection.driver, extra={'size': value['size']})
                    images.append(image)
        return images

    def list_sizes(self, location=None):
        sizes = []
        for value in INSTANCE_TYPES:
            key = value['id']
            size = CloudSigmaNodeSize(id=value['id'], name=value['name'], cpu=value['cpu'], ram=value['memory'], disk=value['disk'], bandwidth=value['bandwidth'], price=self._get_size_price(size_id=key), driver=self.connection.driver)
            sizes.append(size)
        return sizes

    def list_nodes(self):
        response = self.connection.request(action='/servers/info').object
        nodes = []
        for data in response:
            node = self._to_node(data)
            if node:
                nodes.append(node)
        return nodes

    def create_node(self, name, size, image, smp='auto', nic_model='e1000', vnc_password=None, drive_type='hdd'):
        """
        Creates a CloudSigma instance

        @inherits: :class:`NodeDriver.create_node`

        :keyword    name: String with a name for this new node (required)
        :type       name: ``str``

        :keyword    smp: Number of virtual processors or None to calculate
                         based on the cpu speed.
        :type       smp: ``int``

        :keyword    nic_model: e1000, rtl8139 or virtio (is not specified,
                               e1000 is used)
        :type       nic_model: ``str``

        :keyword    vnc_password: If not set, VNC access is disabled.
        :type       vnc_password: ``bool``

        :keyword    drive_type: Drive type (ssd|hdd). Defaults to hdd.
        :type       drive_type: ``str``
        """
        if nic_model not in ['e1000', 'rtl8139', 'virtio']:
            raise CloudSigmaException('Invalid NIC model specified')
        if drive_type not in ['hdd', 'ssd']:
            raise CloudSigmaException('Invalid drive type "%s". Valid types are: hdd, ssd' % drive_type)
        drive_data = {}
        drive_data.update({'name': name, 'size': '%sG' % size.disk, 'driveType': drive_type})
        response = self.connection.request(action='/drives/%s/clone' % image.id, data=dict2str(drive_data), method='POST').object
        if not response:
            raise CloudSigmaException('Drive creation failed')
        drive_uuid = response[0]['drive']
        response = self.connection.request(action='/drives/%s/info' % drive_uuid).object
        imaging_start = time.time()
        while 'imaging' in response[0]:
            response = self.connection.request(action='/drives/%s/info' % drive_uuid).object
            elapsed_time = time.time() - imaging_start
            timed_out = elapsed_time >= self.IMAGING_TIMEOUT
            if 'imaging' in response[0] and timed_out:
                raise CloudSigmaException('Drive imaging timed out')
            time.sleep(1)
        node_data = {}
        node_data.update({'name': name, 'cpu': size.cpu, 'mem': size.ram, 'ide:0:0': drive_uuid, 'boot': 'ide:0:0', 'smp': smp})
        node_data.update({'nic:0:model': nic_model, 'nic:0:dhcp': 'auto'})
        if vnc_password:
            node_data.update({'vnc:ip': 'auto', 'vnc:password': vnc_password})
        response = self.connection.request(action='/servers/create', data=dict2str(node_data), method='POST').object
        if not isinstance(response, list):
            response = [response]
        node = self._to_node(response[0])
        if node is None:
            self.ex_drive_destroy(drive_uuid)
            raise CloudSigmaInsufficientFundsException('Insufficient funds, node creation failed')
        started = self.ex_start_node(node)
        if started:
            node.state = NodeState.RUNNING
        return node

    def ex_destroy_node_and_drives(self, node):
        """
        Destroy a node and all the drives associated with it.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :rtype: ``bool``
        """
        node = self._get_node_info(node)
        drive_uuids = []
        for key, value in node.items():
            if (key.startswith('ide:') or key.startswith('scsi') or key.startswith('block')) and (not (key.endswith(':bytes') or key.endswith(':requests') or key.endswith('media'))):
                drive_uuids.append(value)
        node_destroyed = self.destroy_node(self._to_node(node))
        if not node_destroyed:
            return False
        for drive_uuid in drive_uuids:
            self.ex_drive_destroy(drive_uuid)
        return True

    def ex_static_ip_list(self):
        """
        Return a list of available static IP addresses.

        :rtype: ``list`` of ``str``
        """
        response = self.connection.request(action='/resources/ip/list', method='GET')
        if response.status != 200:
            raise CloudSigmaException('Could not retrieve IP list')
        ips = str2list(response.body)
        return ips

    def ex_drives_list(self):
        """
        Return a list of all the available drives.

        :rtype: ``list`` of ``dict``
        """
        response = self.connection.request(action='/drives/info', method='GET')
        result = str2dicts(response.body)
        return result

    def ex_static_ip_create(self):
        """
        Create a new static IP address.p

        :rtype: ``list`` of ``dict``
        """
        response = self.connection.request(action='/resources/ip/create', method='GET')
        result = str2dicts(response.body)
        return result

    def ex_static_ip_destroy(self, ip_address):
        """
        Destroy a static IP address.

        :param      ip_address: IP address which should be used
        :type       ip_address: ``str``

        :rtype: ``bool``
        """
        response = self.connection.request(action='/resources/ip/%s/destroy' % ip_address, method='GET')
        return response.status == 204

    def ex_drive_destroy(self, drive_uuid):
        """
        Destroy a drive with a specified uuid.
        If the drive is currently mounted an exception is thrown.

        :param      drive_uuid: Drive uuid which should be used
        :type       drive_uuid: ``str``

        :rtype: ``bool``
        """
        response = self.connection.request(action='/drives/%s/destroy' % drive_uuid, method='POST')
        return response.status == 204

    def ex_set_node_configuration(self, node, **kwargs):
        """
        Update a node configuration.
        Changing most of the parameters requires node to be stopped.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :param      kwargs: keyword arguments
        :type       kwargs: ``dict``

        :rtype: ``bool``
        """
        valid_keys = ('^name$', '^parent$', '^cpu$', '^smp$', '^mem$', '^boot$', '^nic:0:model$', '^nic:0:dhcp', '^nic:1:model$', '^nic:1:vlan$', '^nic:1:mac$', '^vnc:ip$', '^vnc:password$', '^vnc:tls', '^ide:[0-1]:[0-1](:media)?$', '^scsi:0:[0-7](:media)?$', '^block:[0-7](:media)?$')
        invalid_keys = []
        keys = list(kwargs.keys())
        for key in keys:
            matches = False
            for regex in valid_keys:
                if re.match(regex, key):
                    matches = True
                    break
            if not matches:
                invalid_keys.append(key)
        if invalid_keys:
            raise CloudSigmaException('Invalid configuration key specified: %s' % ','.join(invalid_keys))
        response = self.connection.request(action='/servers/%s/set' % node.id, data=dict2str(kwargs), method='POST')
        return response.status == 200 and response.body != ''

    def ex_start_node(self, node):
        """
        Start a node.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :rtype: ``bool``
        """
        response = self.connection.request(action='/servers/%s/start' % node.id, method='POST')
        return response.status == 200

    def ex_stop_node(self, node):
        return self.stop_node(node=node)

    def stop_node(self, node):
        """
        Stop (shutdown) a node.

        :param      node: Node which should be used
        :type       node: :class:`libcloud.compute.base.Node`

        :rtype: ``bool``
        """
        response = self.connection.request(action='/servers/%s/stop' % node.id, method='POST')
        return response.status == 204

    def ex_shutdown_node(self, node):
        """
        Stop (shutdown) a node.

        @inherits: :class:`CloudSigmaBaseNodeDriver.ex_stop_node`
        """
        return self.ex_stop_node(node)

    def ex_destroy_drive(self, drive_uuid):
        """
        Destroy a drive.

        :param      drive_uuid: Drive uuid which should be used
        :type       drive_uuid: ``str``

        :rtype: ``bool``
        """
        response = self.connection.request(action='/drives/%s/destroy' % drive_uuid, method='POST')
        return response.status == 204

    def _ex_connection_class_kwargs(self):
        """
        Return the host value based on the user supplied region.
        """
        kwargs = {}
        if not self._host_argument_set:
            kwargs['host'] = API_ENDPOINTS_1_0[self.region]['host']
        return kwargs

    def _to_node(self, data):
        if data:
            try:
                state = self.NODE_STATE_MAP[data['status']]
            except KeyError:
                state = NodeState.UNKNOWN
            if 'server' not in data:
                return None
            public_ips = []
            if 'nic:0:dhcp' in data:
                if isinstance(data['nic:0:dhcp'], list):
                    public_ips = data['nic:0:dhcp']
                else:
                    public_ips = [data['nic:0:dhcp']]
            extra = {}
            extra_keys = [('cpu', 'int'), ('smp', 'auto'), ('mem', 'int'), ('status', 'str')]
            for key, value_type in extra_keys:
                if key in data:
                    value = data[key]
                    if value_type == 'int':
                        value = int(value)
                    elif value_type == 'auto':
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    extra.update({key: value})
            if 'vnc:ip' in data and 'vnc:password' in data:
                extra.update({'vnc_ip': data['vnc:ip'], 'vnc_password': data['vnc:password']})
            node = Node(id=data['server'], name=data['name'], state=state, public_ips=public_ips, private_ips=None, driver=self.connection.driver, extra=extra)
            return node
        return None

    def _get_node(self, node_id):
        nodes = self.list_nodes()
        node = [node for node in nodes if node.id == node.id]
        if not node:
            raise CloudSigmaException('Node with id %s does not exist' % node_id)
        return node[0]

    def _get_node_info(self, node):
        response = self.connection.request(action='/servers/%s/info' % node.id)
        result = str2dicts(response.body)
        return result[0]