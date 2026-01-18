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
class OpenStackNodeDriver(NodeDriver, OpenStackDriverMixin):
    """
    Base OpenStack node driver. Should not be used directly.
    """
    api_name = 'openstack'
    name = 'OpenStack'
    website = 'http://openstack.org/'
    NODE_STATE_MAP = {'BUILD': NodeState.PENDING, 'REBUILD': NodeState.PENDING, 'ACTIVE': NodeState.RUNNING, 'SUSPENDED': NodeState.SUSPENDED, 'SHUTOFF': NodeState.STOPPED, 'DELETED': NodeState.TERMINATED, 'QUEUE_RESIZE': NodeState.PENDING, 'PREP_RESIZE': NodeState.PENDING, 'VERIFY_RESIZE': NodeState.RUNNING, 'PASSWORD': NodeState.PENDING, 'RESCUE': NodeState.PENDING, 'REBOOT': NodeState.REBOOTING, 'RESIZE': NodeState.RECONFIGURING, 'HARD_REBOOT': NodeState.REBOOTING, 'SHARE_IP': NodeState.PENDING, 'SHARE_IP_NO_CONFIG': NodeState.PENDING, 'DELETE_IP': NodeState.PENDING, 'ERROR': NodeState.ERROR, 'UNKNOWN': NodeState.UNKNOWN}
    VOLUME_STATE_MAP = {'creating': StorageVolumeState.CREATING, 'available': StorageVolumeState.AVAILABLE, 'attaching': StorageVolumeState.ATTACHING, 'in-use': StorageVolumeState.INUSE, 'deleting': StorageVolumeState.DELETING, 'error': StorageVolumeState.ERROR, 'error_deleting': StorageVolumeState.ERROR, 'backing-up': StorageVolumeState.BACKUP, 'restoring-backup': StorageVolumeState.BACKUP, 'error_restoring': StorageVolumeState.ERROR, 'error_extending': StorageVolumeState.ERROR}
    SNAPSHOT_STATE_MAP = {'creating': VolumeSnapshotState.CREATING, 'available': VolumeSnapshotState.AVAILABLE, 'deleting': VolumeSnapshotState.DELETING, 'error': VolumeSnapshotState.ERROR, 'restoring': VolumeSnapshotState.RESTORING, 'error_restoring': VolumeSnapshotState.ERROR}

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, **kwargs):
        if cls is OpenStackNodeDriver:
            if api_version == '1.0':
                cls = OpenStack_1_0_NodeDriver
            elif api_version == '1.1':
                cls = OpenStack_1_1_NodeDriver
            elif api_version in ['2.0', '2.1', '2.2']:
                cls = OpenStack_2_NodeDriver
            else:
                raise NotImplementedError('No OpenStackNodeDriver found for API version %s' % api_version)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        OpenStackDriverMixin.__init__(self, **kwargs)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _paginated_request(url, obj, connection, params=None):
        """
        Perform multiple calls in order to have a full list of elements when
        the API responses are paginated.

        :param url: API endpoint
        :type url: ``str``

        :param obj: Result object key
        :type obj: ``str``

        :param connection: The API connection to use to perform the request
        :type connection: ``obj``

        :param params: Any request parameters
        :type params: ``dict``

        :return: ``list`` of API response objects
        :rtype: ``list``
        """
        params = params or {}
        objects = list()
        loop_count = 0
        while True:
            data = connection.request(url, params=params)
            values = data.object.get(obj, list())
            objects.extend(values)
            links = data.object.get('%s_links' % obj, list())
            next_links = [n for n in links if n['rel'] == 'next']
            if next_links:
                next_link = next_links[0]
                query = urlparse.urlparse(next_link['href'])
                params.update(parse_qs(query[4]))
            else:
                break
            loop_count += 1
            if loop_count > PAGINATION_LIMIT:
                raise OpenStackException('Pagination limit reached for %s, the limit is %d. This might indicate that your API is returning a looping next target for pagination!' % (url, PAGINATION_LIMIT), None)
        return {obj: objects}

    def _paginated_request_next(self, path, request_method, response_key):
        """
        Perform multiple calls and retrieve all the elements for a paginated
        response.

        This method utilizes "next" attribute in the response object.

        It also includes an infinite loop protection (if the "next" value
        matches the current path, it will abort).

        :param request_method: Method to call which will send the request and
                               return a response. This method will get passed
                               in "path" as a first argument.

        :param response_key: Key in the response object dictionary which
                             contains actual objects we are interested in.
        """
        iteration_count = 0
        result = []
        while path:
            response = request_method(path)
            items = response.object.get(response_key, []) or []
            result.extend(items)
            next_path = response.object.get('next', None)
            if next_path == path:
                break
            if iteration_count > PAGINATION_LIMIT:
                raise OpenStackException('Pagination limit reached for %s, the limit is %d. This might indicate that your API is returning a looping next target for pagination!' % (path, PAGINATION_LIMIT), None)
            path = next_path
            iteration_count += 1
        return result

    def destroy_node(self, node):
        uri = '/servers/%s' % node.id
        resp = self.connection.request(uri, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def reboot_node(self, node):
        return self._reboot_node(node, reboot_type='HARD')

    def start_node(self, node):
        return self._post_simple_node_action(node, 'os-start')

    def stop_node(self, node):
        return self._post_simple_node_action(node, 'os-stop')

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
        return self._to_nodes(self.connection.request('/servers/detail', params=params).object)

    def create_volume(self, size, name, location=None, snapshot=None, ex_volume_type=None):
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

        :return: The newly created volume.
        :rtype: :class:`StorageVolume`
        """
        volume = {'display_name': name, 'display_description': name, 'size': size, 'metadata': {'contents': name}}
        if ex_volume_type:
            volume['volume_type'] = ex_volume_type
        if location:
            volume['availability_zone'] = location
        if snapshot:
            volume['snapshot_id'] = snapshot.id
        resp = self.connection.request('/os-volumes', method='POST', data={'volume': volume})
        return self._to_volume(resp.object)

    def destroy_volume(self, volume):
        return self.connection.request('/os-volumes/%s' % volume.id, method='DELETE').success()

    def attach_volume(self, node, volume, device='auto'):
        if device == 'auto':
            device = None
        return self.connection.request('/servers/%s/os-volume_attachments' % node.id, method='POST', data={'volumeAttachment': {'volumeId': volume.id, 'device': device}}).success()

    def detach_volume(self, volume, ex_node=None):
        failed_nodes = []
        for attachment in volume.extra['attachments']:
            if not ex_node or ex_node.id in filter(None, (attachment.get('serverId'), attachment.get('server_id'))):
                response = self.connection.request('/servers/%s/os-volume_attachments/%s' % (attachment.get('serverId') or attachment['server_id'], attachment['id']), method='DELETE')
                if not response.success():
                    failed_nodes.append(attachment.get('serverId') or attachment['server_id'])
        if failed_nodes:
            raise OpenStackException('detach_volume failed for nodes with id: %s' % ', '.join(failed_nodes), 500, self)
        return True

    def list_volumes(self):
        return self._to_volumes(self.connection.request('/os-volumes').object)

    def ex_get_volume(self, volumeId):
        return self._to_volume(self.connection.request('/os-volumes/%s' % volumeId).object)

    def list_images(self, location=None, ex_only_active=True):
        """
        Lists all active images

        @inherits: :class:`NodeDriver.list_images`

        :param ex_only_active: True if list only active (optional)
        :type ex_only_active: ``bool``

        """
        return self._to_images(self.connection.request('/images/detail').object, ex_only_active)

    def get_image(self, image_id):
        """
        Get an image based on an image_id

        @inherits: :class:`NodeDriver.get_image`

        :param image_id: Image identifier
        :type image_id: ``str``

        :return: A NodeImage object
        :rtype: :class:`NodeImage`

        """
        return self._to_image(self.connection.request('/images/{}'.format(image_id)).object['image'])

    def list_sizes(self, location=None):
        return self._to_sizes(self.connection.request('/flavors/detail').object)

    def list_locations(self):
        return [NodeLocation(0, '', '', self)]

    def _ex_connection_class_kwargs(self):
        return self.openstack_connection_kwargs()

    def ex_get_node_details(self, node_id):
        """
        Lists details of the specified server.

        :param       node_id: ID of the node which should be used
        :type        node_id: ``str``

        :rtype: :class:`Node`
        """
        if isinstance(node_id, Node):
            node_id = node_id.id
        uri = '/servers/%s' % node_id
        try:
            resp = self.connection.request(uri, method='GET')
        except BaseHTTPError as e:
            if e.code == httplib.NOT_FOUND:
                return None
            raise
        return self._to_node_from_obj(resp.object)

    def ex_soft_reboot_node(self, node):
        """
        Soft reboots the specified server

        :param      node:  node
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        return self._reboot_node(node, reboot_type='SOFT')

    def ex_hard_reboot_node(self, node):
        """
        Hard reboots the specified server

        :param      node:  node
        :type       node: :class:`Node`

        :rtype: ``bool``
        """
        return self._reboot_node(node, reboot_type='HARD')