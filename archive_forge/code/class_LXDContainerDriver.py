import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
class LXDContainerDriver(ContainerDriver):
    """
    Driver for LXD REST API of LXC containers
    https://lxd.readthedocs.io/en/stable-2.0/rest-api/
    https://github.com/lxc/lxd/blob/master/doc/rest-api.md
    """
    type = Provider.LXD
    name = 'LXD'
    website = 'https://linuxcontainers.org/'
    connectionCls = LXDConnection
    supports_clusters = False
    version = '1.0'
    default_time_out = 30
    default_architecture = ''
    default_profiles = 'default'
    default_ephemeral = False

    def __init__(self, key='', secret='', secure=False, host='localhost', port=8443, key_file=None, cert_file=None, ca_cert=None, certificate_validator=check_certificates):
        if key_file:
            if not cert_file:
                raise LXDAPIException(message='Need both private key and certificate files for tls authentication')
            self.connectionCls = LXDtlsConnection
            self.key_file = key_file
            self.cert_file = cert_file
            self.certificate_validator = certificate_validator
            secure = True
        if host.startswith('https://'):
            secure = True
        host = strip_http_prefix(host=host)
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, key_file=key_file, cert_file=cert_file)
        if ca_cert:
            self.connection.connection.ca_cert = ca_cert
        else:
            self.connection.connection.ca_cert = False
        self.connection.secure = secure
        self.connection.host = host
        self.connection.port = port
        self.version = self._get_api_version()

    def build_operation_websocket_url(self, uuid, w_secret):
        uri = 'wss://%s:%s/%s/operations/%s/websocket?secret=%s' % (self.connection.host, self.connection.port, self.version, uuid, w_secret)
        return uri

    def ex_get_api_endpoints(self):
        """
        Description: List of supported APIs
        Authentication: guest
        Operation: sync
        Return: list of supported API endpoint URLs

        """
        response = self.connection.request('/')
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return response_dict['metadata']

    def ex_get_server_configuration(self):
        """

        Description: Server configuration and environment information
        Authentication: guest, untrusted or trusted
        Operation: sync
        Return: Dict representing server state

        The returned configuration depends on whether the connection
        is trusted or not
        :rtype: :class: .LXDServerInfo

        """
        response = self.connection.request('/%s' % self.version)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        meta = response_dict['metadata']
        return LXDServerInfo.build_from_response(metadata=meta)

    def deploy_container(self, name, image, cluster=None, parameters=None, start=True, ex_architecture=default_architecture, ex_profiles=None, ex_ephemeral=default_ephemeral, ex_config=None, ex_devices=None, ex_instance_type=None, ex_timeout=default_time_out):
        """
        Create a new container
        Authentication: trusted
        Operation: async

        :param name: The name of the new container.
        64 chars max, ASCII, no slash, no colon and no comma
        :type  name: ``str``

        :param image: The container image to deploy. Currently not used
        :type  image: :class:`.ContainerImage`

        :param cluster: The cluster to deploy to, None is default
        :type  cluster: :class:`.ContainerCluster`

        :param parameters: Container Image parameters.
        This parameter should represent the
        the ``source`` dictionary expected by the  LXD API call. For more
        information how this parameter should be structured see
        https://github.com/lxc/lxd/blob/master/doc/rest-api.md
        :type  parameters: ``str``

        :param start: Start the container on deployment. True is the default
        :type  start: ``bool``

        :param ex_architecture: string e.g. x86_64
        :type  ex_architecture: ``str``

        :param ex_profiles: List of profiles
        :type  ex_profiles: ``list``

        :param ex_ephemeral: Whether to destroy the container on shutdown
        :type  ex_ephemeral: ``bool``

        :param ex_config: Config override e.g.  {"limits.cpu": "2"}
        :type  ex_config: ``dict``

        :param ex_devices: optional list of devices the container should have
        :type  ex_devices: ``dict``

        :param ex_instance_type: An optional instance type
        to use as basis for limits e.g. "c2.micro"
        :type  ex_instance_type: ``str``

        :param ex_timeout: Timeout
        :type  ex_timeout: ``int``

        :rtype: :class:`libcloud.container.base.Container`
        """
        if parameters:
            parameters = json.loads(parameters)
            if parameters['source'].get('mode', None) == 'pull':
                try:
                    image = self.install_image(path=None, ex_timeout=ex_timeout, **parameters)
                except Exception as e:
                    raise LXDAPIException(message='Deploying container failed:  Image could not be installed. %r' % e)
                parameters = {'source': {'type': 'image', 'fingerprint': image.extra['fingerprint']}}
        cont_params = LXDContainerDriver._fix_cont_params(architecture=ex_architecture, profiles=ex_profiles, ephemeral=ex_ephemeral, config=ex_config, devices=ex_devices, instance_type=ex_instance_type)
        container = self._deploy_container_from_image(name=name, image=image, parameters=parameters, cont_params=cont_params, timeout=ex_timeout)
        if start:
            container.start()
        return container

    def get_container(self, id, ex_get_ip_addr=True):
        """
        Get a container by ID

        :param id: The ID of the container to get
        :type  id: ``str``

        :param ex_get_ip_addr: Indicates whether ip addresses
        should also be included. This requires an extra GET request
        :type  ex_get_ip_addr: ``boolean```

        :rtype: :class:`libcloud.container.base.Container`
        """
        req = '/{}/containers/{}'.format(self.version, id)
        response = self.connection.request(req)
        result_dict = response.parse_body()
        assert_response(response_dict=result_dict, status_code=200)
        metadata = result_dict['metadata']
        ips = []
        if ex_get_ip_addr:
            req = '/{}/containers/{}/state'.format(self.version, id)
            ip_response = self.connection.request(req)
            ip_result_dict = ip_response.parse_body()
            assert_response(response_dict=ip_result_dict, status_code=200)
            if ip_result_dict['metadata']['network'] is not None:
                networks = ip_result_dict['metadata']['network']['eth0']
                addresses = networks['addresses']
                for item in addresses:
                    ips.append(item['address'])
        metadata.update({'ips': ips})
        return self._to_container(metadata=metadata)

    def start_container(self, container, ex_timeout=default_time_out, ex_force=True, ex_stateful=True):
        """
        Start a container

        :param container: The container to start
        :type  container: :class:`libcloud.container.base.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :param ex_force:
        :type  ex_force: ``boolean``

        :param ex_stateful:
        :type  ex_stateful: ``boolean``

        :rtype: :class:`libcloud.container.base.Container`
        """
        return self._do_container_action(container=container, action='start', timeout=ex_timeout, force=ex_force, stateful=ex_stateful)

    def stop_container(self, container, ex_timeout=default_time_out, ex_force=True, ex_stateful=True):
        """
        Stop the given container

        :param container: The container to be stopped
        :type  container: :class:`libcloud.container.base.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :param ex_force:
        :type  ex_force: ``boolean``

        :param ex_stateful:
        :type  ex_stateful: ``boolean``

        :rtype: :class:`libcloud.container.base.Container
        """
        return self._do_container_action(container=container, action='stop', timeout=ex_timeout, force=ex_force, stateful=ex_stateful)

    def restart_container(self, container, ex_timeout=default_time_out, ex_force=True, ex_stateful=True):
        """
        Restart a deployed container

        :param container: The container to restart
        :type  container: :class:`.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :param ex_force:
        :type  ex_force: ``boolean``

        :param ex_stateful:
        :type  ex_stateful: ``boolean``

        :rtype: :class:`libcloud.container.base.Container
        """
        return self._do_container_action(container=container, action='restart', timeout=ex_timeout, force=ex_force, stateful=ex_stateful)

    def ex_freeze_container(self, container, ex_timeout=default_time_out):
        """
        Set the given container into a freeze state

        :param container: The container to restart
        :type  container: :class:`.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :rtype :class: `libcloud.container.base.Container
        """
        return self._do_container_action(container=container, action='freeze', timeout=ex_timeout, force=True, stateful=True)

    def ex_unfreeze_container(self, container, ex_timeout=default_time_out):
        """
        Set the given container into  unfreeze state

        :param container: The container to restart
        :type  container: :class:`.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :rtype :class: `libcloud.container.base.Container
        """
        return self._do_container_action(container=container, action='unfreeze', timeout=ex_timeout, force=True, stateful=True)

    def destroy_container(self, container, ex_timeout=default_time_out):
        """
        Destroy a deployed container. Raises and exception

        if the container is running

        :param container: The container to destroy
        :type  container: :class:`.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout ``int``

        :rtype: :class:`libcloud.container.base.Container
        """
        req = '/{}/containers/{}'.format(self.version, container.name)
        try:
            response = self.connection.request(req, method='DELETE')
            response_dict = response.parse_body()
            assert_response(response_dict=response_dict, status_code=100)
        except BaseHTTPError as err:
            raise self._get_lxd_api_exception_for_error(err)
        try:
            id = response_dict['metadata']['id']
            req = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, ex_timeout)
            response = self.connection.request(req)
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message != 'not found':
                raise lxd_exception
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        container = Container(driver=self, name=container.name, id=container.name, state=ContainerState.TERMINATED, image=None, ip_addresses=[], extra=None)
        return container

    def ex_execute_cmd_on_container(self, cont_id, command, **config):
        """
        Description: run a remote command
        Operation: async

        Return: Depends on the  the configuration

        if wait-for-websocket=true and interactive=false
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=fds["0"],
            secret_1=fds["1"],
            secret_2=fds["2"],
            control=fds["control"],
            output={}, result=None

        if wait-for-websocket=true and interactive=true
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=fds["0"],
            secret_1=None,
            secret_2=None,
            control=fds["control"],
            output={}, result=None

        if interactive=false and record-output=true
        returns a LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=None,
            secret_1=None,
            secret_2=None,
            control=None,
            output=output, result=result

        if none of the above it assumes that the command has
        been executed and returns LXDContainerExecuteResult with:
            uuid=uuid,
            secret_0=None,
            secret_1=None,
            secret_2=None,
            control=None,
            output=None, result=result


        in all the above uuid is the operation id

        :param cont_id: The container name to run the commands
        ":type cont_id: ``str``

        :param command: a list of strings indicating the commands
        and their arguments e.g: ["/bin/bash ls -l"]
        :type  command ``list``

        :param config: Dict with extra arguments.

            For example:

            width:  Initial width of the terminal default 80
            height: Initial height of the terminal default 25
            user:   User to run the command as default 1000
            group: Group to run the  command as default 1000
            cwd: Current working directory default /tmp

            wait-for-websocket: Whether to wait for a connection
            before starting the process. Default False

            record-output: Whether to store stdout and stderr
            (only valid with wait-for-websocket=false)
            (requires API extension container_exec_recording). Default False

            interactive: Whether to allocate a pts device
            instead of PIPEs. Default true

        :type config ``dict``

        :rtype LXDContainerExecuteResult
        """
        input = {'command': command}
        input = LXDContainerDriver._create_exec_configuration(input, **config)
        data = json.dumps(input)
        req = '/{}/containers/{}/exec'.format(self.version, cont_id)
        response = self.connection.request(req, method='POST', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=100)
        fds = response_dict['metadata']['metadata']['fds']
        uuid = response_dict['metadata']['id']
        if input['wait-for-websocket'] is True and input['interactive'] is False:
            return LXDContainerExecuteResult(uuid=uuid, secret_0=fds['0'], secret_1=fds['1'], secret_2=fds['2'], control=fds['control'], output={}, result=None)
        elif input['wait-for-websocket'] is True and input['interactive'] is True:
            return LXDContainerExecuteResult(uuid=uuid, secret_0=fds['0'], secret_1=None, secret_2=None, control=fds['control'], output={}, result=None)
        elif input['interactive'] is False and input['record-output'] is True:
            output = response_dict['metadata']['metadata']['output']
            result = response_dict['metadata']['metadata']['result']
            return LXDContainerExecuteResult(uuid=uuid, secret_0=None, secret_1=None, secret_2=None, control=None, output=output, result=result)
        else:
            result = response_dict['metadata']['metadata']['result']
            return LXDContainerExecuteResult(uuid=uuid, secret_0=None, secret_1=None, secret_2=None, control=None, output={}, result=result)

    def list_containers(self, image=None, cluster=None, ex_detailed=True):
        """
        List the deployed container images

        :param image: Filter to containers with a certain image
        :type  image: :class:`.ContainerImage`

        :param cluster: Filter to containers in a cluster
        :type  cluster: :class:`.ContainerCluster`

        :param ex_detailed: Flag indicating whether detail info
        of the containers is required. This will cause a
        GET request for every container present in the
        host. Default is True
        :type ex_detailed: ``bool``

        :rtype: ``list`` of :class:`libcloud.container.base.Container
        """
        result = self.connection.request('/%s/containers' % self.version)
        result = result.parse_body()
        assert_response(response_dict=result, status_code=200)
        meta = result['metadata']
        containers = []
        for item in meta:
            container_id = item.split('/')[-1]
            if not ex_detailed:
                container = Container(driver=self, name=container_id, state=ContainerState.UNKNOWN, id=container_id, image=image, ip_addresses=[], extra={})
            else:
                container = self.get_container(id=container_id)
            containers.append(container)
        return containers

    def ex_get_image(self, fingerprint):
        """
        Returns a container image from the given image fingerprint

        :param fingerprint: image fingerprint
        :type  fingerprint: ``str``

        :rtype: :class:`.ContainerImage`
        """
        req = '/{}/images/{}'.format(self.version, fingerprint)
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self._to_image(metadata=response_dict['metadata'])

    def install_image(self, path, ex_timeout=default_time_out, **ex_img_data):
        """
        Install a container image from a remote path. Not that the
        path currently is not used. Image data should be provided
        under the key 'ex_img_data'. Creating an image in LXD is an
        asynchronous operation

        :param path: Path to the container image
        :type  path: ``str``

        :param ex_timeout: Time to wait before signaling timeout
        :type  ex_timeout: ``int``

        :param ex_img_data: Dictionary describing the image data
        :type  ex_img_data: ``dict``

        :rtype: :class:`.ContainerImage`
        """
        if not ex_img_data:
            msg = 'Install an image for LXD requires specification of image_data'
            raise LXDAPIException(message=msg)
        data = ex_img_data['source']
        config = {'public': data.get('public', True), 'auto_update': data.get('auto_update', False), 'aliases': [data.get('aliases', {})], 'source': {'type': 'url', 'mode': 'pull', 'url': data['url']}}
        config = json.dumps(config)
        response = self.connection.request('/%s/images' % self.version, method='POST', data=config)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=100)
        try:
            id = response_dict['metadata']['id']
            req = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, ex_timeout)
            response = self.connection.request(req)
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message != 'not found':
                raise lxd_exception
        config = json.loads(config)
        if len(config['aliases']) != 0 and 'name' in config['aliases'][0]:
            image_alias = config['aliases'][0]['name']
        else:
            image_alias = config['source']['url'].split('/')[-1]
        has, fingerprint = self.ex_has_image(alias=image_alias)
        if not has:
            raise LXDAPIException(message='Image %s was not installed ' % image_alias)
        return self.ex_get_image(fingerprint=fingerprint)

    def list_images(self):
        """
        List of URLs for images the server is publishing

        :rtype: ``list`` of :class:`.ContainerImage`
        """
        response = self.connection.request('/%s/images' % self.version)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        metadata = response_dict['metadata']
        images = []
        for image in metadata:
            fingerprint = image.split('/')[-1]
            images.append(self.ex_get_image(fingerprint=fingerprint))
        return images

    def ex_has_image(self, alias):
        """
        Helper function. Returns true and the image fingerprint
        if the image with the given alias exists on the host.

        :param alias: the image alias
        :type  alias: ``str``

        :rtype:  ``tupple`` :: (``boolean``, ``str``)
        """
        try:
            response = self.connection.request('/{}/images/aliases/{}'.format(self.version, alias))
            metadata = response.object['metadata']
            return (True, metadata.get('target'))
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message == 'not found':
                return (False, -1)
            else:
                raise lxd_exception
        except Exception as err:
            raise self._get_lxd_api_exception_for_error(err)

    def ex_list_storage_pools(self, detailed=True):
        """
        Returns a list of storage pools defined currently defined on the host

        Description: list of storage pools
        Authentication: trusted
        Operation: sync

        ":rtype: list of StoragePool items
        """
        response = self.connection.request('/%s/storage-pools' % self.version)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        pools = []
        for pool_item in response_dict['metadata']:
            pool_name = pool_item.split('/')[-1]
            if not detailed:
                pools.append(self._to_storage_pool({'name': pool_name, 'driver': None, 'used_by': None, 'config': None, 'managed': None}))
            else:
                pools.append(self.ex_get_storage_pool(id=pool_name))
        return pools

    def ex_get_storage_pool(self, id):
        """
        Returns  information about a storage pool
        :param id: the name of the storage pool
        :rtype: :class: StoragePool
        """
        req = '/{}/storage-pools/{}'.format(self.version, id)
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        if not response_dict['metadata']:
            msg = 'Storage pool with name {} has no data'.format(id)
            raise LXDAPIException(message=msg)
        return self._to_storage_pool(data=response_dict['metadata'])

    def ex_create_storage_pool(self, definition):
        """
        Create a storage_pool from definition.

        Implements POST /1.0/storage-pools

        The `definition` parameter defines
        what the storage pool will be.  An
        example config for the zfs driver is:

                   {
                       "config": {
                           "size": "10GB"
                       },
                       "driver": "zfs",
                       "name": "pool1"
                   }

        Note that **all** fields in the `definition` parameter are strings.
        Note that size has to be at least 64MB in order to create the pool

        For further details on the storage pool types see:
        https://lxd.readthedocs.io/en/latest/storage/

        The function returns the a `StoragePool` instance, if it is
        successfully created, otherwise an LXDAPIException is raised.

        :param definition: the fields to pass to the LXD API endpoint
        :type definition: dict

        :returns: a storage pool if successful,
        raises NotFound if not found
        :rtype: :class:`StoragePool`

        :raises: :class:`LXDAPIExtensionNotAvailable`
        if the 'storage' api extension is missing.
        :raises: :class:`LXDAPIException`
        if the storage pool couldn't be created.
        """
        if not definition:
            raise LXDAPIException('Cannot create a storage pool  without a definition')
        data = json.dumps(definition)
        response = self.connection.request('/%s/storage-pools' % self.version, method='POST', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self.ex_get_storage_pool(id=definition['name'])

    def ex_delete_storage_pool(self, id):
        """Delete the storage pool.

        Implements DELETE /1.0/storage-pools/<self.name>

        Deleting a storage pool may fail if it is being used.  See the LXD
        documentation for further details.

        :raises: :class:`LXDAPIException` if the storage pool can't be deleted.
        """
        req = '/{}/storage-pools/{}'.format(self.version, id)
        response = self.connection.request(req, method='DELETE')
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)

    def ex_list_storage_pool_volumes(self, pool_id, detailed=True):
        """
        Description: list of storage volumes
        associated with the given storage pool

        :param pool_id: the id of the storage pool to query
        :param detailed: boolean flag.
        If True extra API calls are made to fill in the missing details
                                       of the storage volumes

        Authentication: trusted
        Operation: sync
        Return: list of storage volumes that
        currently exist on a given storage pool

        :rtype: A list of :class: StorageVolume
        """
        req = '/{}/storage-pools/{}/volumes'.format(self.version, pool_id)
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        volumes = []
        for volume in response_dict['metadata']:
            volume = volume.split('/')
            name = volume[-1]
            type = volume[-2]
            if not detailed:
                metadata = {'config': {'size': None}, 'name': name, 'type': type, 'used_by': None}
                volumes.append(self._to_storage_volume(pool_id=pool_id, metadata=metadata))
            else:
                volume = self.ex_get_storage_pool_volume(pool_id=pool_id, type=type, name=name)
                volumes.append(volume)
        return volumes

    def ex_get_storage_pool_volume(self, pool_id, type, name):
        """
        Description: information about a storage volume
        of a given type on a storage pool
        Introduced: with API extension storage
        Authentication: trusted
        Operation: sync
        Return: A StorageVolume  representing a storage volume
        """
        req = '/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name)
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self._to_storage_volume(pool_id=pool_id, metadata=response_dict['metadata'])

    def ex_get_volume_by_name(self, name, vol_type='custom'):
        """
        Returns a storage volume that has the given name.
        The function will loop over all storage-polls available
        and will pick the first volume from the first storage poll
        that matches the given name. Thus this function can be
        quite expensive

        :param name: The name of the volume to look for
        :type  name: str

        :param vol_type: The type of the volume default is custom
        :type  vol_type: str

        :return: A StorageVolume  representing a storage volume
        """
        req = '/%s/storage-pools' % self.version
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        pools = response_dict['metadata']
        for pool in pools:
            pool_id = pool.split('/')[-1]
            volumes = self.ex_list_storage_pool_volumes(pool_id=pool_id)
            for vol in volumes:
                if vol.name == name:
                    return vol
        return None

    def create_volume(self, pool_id, definition, **kwargs):
        """
        Create a new storage volume on a given storage pool
        Operation: sync or async (when copying an existing volume)
        :return: A StorageVolume  representing a storage volume
        """
        if not definition:
            raise LXDAPIException('Cannot create a storage volume without a definition')
        size_type = definition.pop('size_type')
        definition['config']['size'] = str(LXDContainerDriver._to_bytes(definition['config']['size'], size_type=size_type))
        data = json.dumps(definition)
        req = '/{}/storage-pools/{}/volumes'.format(self.version, pool_id)
        response = self.connection.request(req, method='POST', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self.ex_get_storage_pool_volume(pool_id=pool_id, type=definition['type'], name=definition['name'])

    def attach_volume(self, container_id, volume_id, pool_id, name, path, ex_timeout=default_time_out):
        """
        Attach the volume with id volume_id
        to the container with id container_id
        """
        container = self.get_container(id=container_id)
        config = container.extra
        config['devices'] = {name: {'path': path, 'type': 'disk', 'source': volume_id, 'pool': pool_id}}
        data = json.dumps(config)
        req = '/{}/containers/{}'.format(self.version, container_id)
        response = self.connection.request(req, method='PUT', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=100)
        try:
            oid = response_dict['metadata']['id']
            req = '/{}/operations/{}/wait?timeout={}'.format(self.version, oid, ex_timeout)
            response = self.connection.request(req)
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message != 'not found':
                raise lxd_exception
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self.get_container(id=container_id, ex_get_ip_addr=True)

    def ex_replace_storage_volume_config(self, pool_id, type, name, definition):
        """
        Replace the storage volume information
        :param pool_id:
        :param type:
        :param name:
        :param definition
        """
        if not definition:
            raise LXDAPIException('Cannot create a storage volume without a definition')
        data = json.dumps(definition)
        response = self.connection.request('/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name), method='PUT', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self.ex_get_storage_pool_volume(pool_id=pool_id, type=type, name=name)

    def ex_delete_storage_pool_volume(self, pool_id, type, name):
        """
        Delete a storage volume of a given type on a given storage pool

        :param pool_id:
        :type ``str``

        :param type:
        :type  ``str``

        :param name:
        :type ``str``

        :return:
        """
        try:
            req = '/{}/storage-pools/{}/volumes/{}/{}'.format(self.version, pool_id, type, name)
            response = self.connection.request(req, method='DELETE')
            response_dict = response.parse_body()
            assert_response(response_dict=response_dict, status_code=200)
        except BaseHTTPError as err:
            raise self._get_lxd_api_exception_for_error(err)
        return True

    def ex_list_networks(self):
        """
        Returns a list of networks.
        Implements GET /1.0/networks
        Authentication: trusted
        Operation: sync

        :rtype: list of LXDNetwork objects
        """
        req = '/%s/networks' % self.version
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        nets = response_dict['metadata']
        networks = []
        for net in nets:
            name = net.split('/')[-1]
            networks.append(self.ex_get_network(name=name))
        return networks

    def ex_get_network(self, name):
        """
        Returns the LXD network with the given name.
        Implements GET /1.0/networks/<name>

        Authentication: trusted
        Operation: sync

        :param name: The name of the network to return
        :type  name: str

        :rtype: LXDNetwork
        """
        req = '/{}/networks/{}'.format(self.version, name)
        response = self.connection.request(req)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return LXDNetwork.build_from_response(response_dict['metadata'])

    def ex_create_network(self, name, **kwargs):
        """
        Create a new network with the given name and
        and the specified configuration

        Authentication: trusted
        Operation: sync

        :param name: The name of the new network
        :type  name: str
        """
        kwargs['name'] = name
        data = json.dumps(kwargs)
        req = '/%s/networks' % self.version
        response = self.connection.request(req, method='POST', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return self.ex_get_network(name=name)

    def ex_delete_network(self, name):
        """
        Delete the network with the given name
        Authentication: trusted
        Operation: sync

        :param name: The network name to delete
        :type  name: str

        :return: True is successfully deleted the network
        """
        req = '/{}/networks/{}'.format(self.version, name)
        response = self.connection.request(req, method='DELETE')
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=200)
        return True

    def _to_container(self, metadata):
        """
        Returns Container instance built from the given metadata

        :param metadata: dictionary with the container metadata
        :type  metadata: ``dict``

        :rtype :class:`libcloud.container.base.Container
        """
        name = metadata['name']
        state = metadata['status']
        if state == 'Running':
            state = ContainerState.RUNNING
        elif state == 'Frozen':
            state = ContainerState.PAUSED
        else:
            state = ContainerState.STOPPED
        extra = metadata
        img_id = metadata['config'].get('volatile.base_image', None)
        img_version = metadata['config'].get('image.version', None)
        ips = metadata['ips']
        image = ContainerImage(id=img_id, name=img_id, path=None, version=img_version, driver=self, extra=None)
        container = Container(driver=self, name=name, id=name, state=state, image=image, ip_addresses=ips, extra=extra)
        return container

    def _do_container_action(self, container, action, timeout, force, stateful):
        """
        change the container state by performing the given action
        action may be either stop, start, restart, freeze or unfreeze
        """
        if action not in LXD_API_STATE_ACTIONS:
            raise ValueError('Invalid action specified')
        state = container.state
        data = {'action': action, 'timeout': timeout}
        data = json.dumps(data)
        req = '/{}/containers/{}/state'.format(self.version, container.name)
        response = self.connection.request(req, method='PUT', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=100)
        if not timeout:
            timeout = LXDContainerDriver.default_time_out
        try:
            id = response_dict['metadata']['id']
            req = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, timeout)
            response = self.connection.request(req)
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message != 'not found':
                raise lxd_exception
        if state == ContainerState.RUNNING and container.extra['ephemeral'] and (action == 'stop'):
            container = Container(driver=self, name=container.name, id=container.name, state=ContainerState.TERMINATED, image=None, ip_addresses=[], extra=None)
            return container
        return self.get_container(id=container.name)

    def _to_image(self, metadata):
        """
        Returns a container image from the given metadata

        :param metadata:
        :type  metadata: ``dict``

        :rtype: :class:`.ContainerImage`
        """
        fingerprint = metadata.get('fingerprint')
        aliases = metadata.get('aliases', [])
        if aliases:
            name = metadata.get('aliases')[0].get('name')
        else:
            name = metadata.get('properties', {}).get('description') or fingerprint
        version = metadata.get('update_source', {}).get('alias')
        extra = metadata
        return ContainerImage(id=fingerprint, name=name, path=None, version=version, driver=self, extra=extra)

    def _to_storage_pool(self, data):
        """
        Given a dictionary with the storage pool configuration
        it returns a StoragePool object
        :param data: the storage pool configuration
        :return: :class: .StoragePool
        """
        return LXDStoragePool(name=data['name'], driver=data['driver'], used_by=data['used_by'], config=['config'], managed=False)

    def _deploy_container_from_image(self, name, image, parameters, cont_params, timeout=default_time_out):
        """
        Deploy a new container from the given image

        :param name: the name of the container
        :param image: .ContainerImage

        :param parameters: dictionary describing the source attribute
        :type  parameters ``dict``

        :param cont_params: dictionary describing the container configuration
        :type  cont_params: dict

        :param timeout: Time to wait for the operation before timeout
        :type  timeout: int

        :rtype: :class: .Container
        """
        if cont_params is None:
            raise LXDAPIException(message='cont_params must be a valid dict')
        data = {'name': name, 'source': {'type': 'none'}}
        if parameters:
            data['source'].update(parameters['source'])
        if data['source']['type'] not in LXD_API_IMAGE_SOURCE_TYPE:
            msg = 'source type must in ' + str(LXD_API_IMAGE_SOURCE_TYPE)
            raise LXDAPIException(message=msg)
        data.update(cont_params)
        data = json.dumps(data)
        response = self.connection.request('/%s/containers' % self.version, method='POST', data=data)
        response_dict = response.parse_body()
        assert_response(response_dict=response_dict, status_code=100)
        if not timeout:
            timeout = LXDContainerDriver.default_time_out
        try:
            id = response_dict['metadata']['id']
            req_str = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, timeout)
            response = self.connection.request(req_str)
        except BaseHTTPError as err:
            lxd_exception = self._get_lxd_api_exception_for_error(err)
            if lxd_exception.message != 'not found':
                raise lxd_exception
        return self.get_container(id=name)

    def _to_storage_volume(self, pool_id, metadata):
        """
        Returns StorageVolume object from metadata
        :param metadata: dict representing the volume
        :rtype: StorageVolume
        """
        size = 0
        if 'size' in metadata['config'].keys():
            size = LXDContainerDriver._to_gb(metadata['config'].pop('size'))
        extra = {'pool_id': pool_id, 'type': metadata['type'], 'used_by': metadata['used_by'], 'config': metadata['config']}
        return StorageVolume(id=metadata['name'], name=metadata['name'], driver=self, size=size, extra=extra)

    def _get_api_version(self):
        """
        Get the LXD API version
        """
        return LXDContainerDriver.version

    def _ex_connection_class_kwargs(self):
        """
        Return extra connection keyword arguments which are passed to the
        Connection class constructor.
        """
        if hasattr(self, 'key_file') and hasattr(self, 'cert_file'):
            return {'key_file': self.key_file, 'cert_file': self.cert_file, 'certificate_validator': self.certificate_validator}
        return super()._ex_connection_class_kwargs()

    @staticmethod
    def _create_exec_configuration(input, **config):
        """
        Prepares the input parameters for executyion API call
        """
        if 'environment' in config.keys():
            input['environment'] = config['environment']
        if 'width' in config.keys():
            input['width'] = int(config['width'])
        else:
            input['width'] = 80
        if 'height' in config.keys():
            input['height'] = int(config['height'])
        else:
            input['height'] = 25
        if 'user' in config.keys():
            input['user'] = config['user']
        if 'group' in config.keys():
            input['group'] = config['group']
        if 'cwd' in config.keys():
            input['cwd'] = config['cwd']
        if 'wait-for-websocket' in config.keys():
            input['wait-for-websocket'] = config['wait-for-websocket']
        else:
            input['wait-for-websocket'] = False
        if 'record-output' in config.keys():
            input['record-output'] = config['record-output']
        if 'interactive' in config.keys():
            input['interactive'] = config['interactive']
        return input

    @staticmethod
    def _fix_cont_params(architecture, profiles, ephemeral, config, devices, instance_type):
        """
        Returns a dict with the container parameters
        """
        cont_params = {}
        if architecture is not None:
            cont_params['architecture'] = architecture
        if profiles is not None:
            cont_params['profiles'] = profiles
        else:
            cont_params['profiles'] = [LXDContainerDriver.default_profiles]
        if ephemeral is not None:
            cont_params['ephemeral'] = ephemeral
        else:
            cont_params['ephemeral'] = LXDContainerDriver.default_ephemeral
        if config is not None:
            cont_params['config'] = config
        if devices is not None:
            cont_params['devices'] = devices
        if instance_type is not None:
            cont_params['instance_type'] = instance_type
        return cont_params

    def _get_lxd_api_exception_for_error(self, error):
        error_dict = json.loads(error.message)
        message = error_dict.get('error')
        return LXDAPIException(message=message, error_type=error_dict.get('type', ''))

    @staticmethod
    def _to_gb(size):
        """
        Convert the given size in bytes to gigabyte
        :param size: in bytes
        :return: int representing the gigabytes
        """
        size = int(size)
        return size // 10 ** 9

    @staticmethod
    def _to_bytes(size, size_type='GB'):
        """
        convert the given size in GB to bytes
        :param size: in GBs
        :return: int representing bytes
        """
        size = int(size)
        if size_type == 'GB':
            return size * 10 ** 9
        elif size_type == 'MB':
            return size * 10 ** 6