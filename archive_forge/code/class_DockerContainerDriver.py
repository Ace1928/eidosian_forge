import os
import re
import shlex
import base64
import datetime
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
class DockerContainerDriver(ContainerDriver):
    """
    Docker container driver class.

    >>> from libcloud.container.providers import get_driver
    >>> driver = get_driver('docker')
    >>> conn = driver(host='198.61.239.128', port=4243)
    >>> conn.list_containers()
    or connecting to http basic auth protected https host:
    >>> conn = driver('user', 'pass', host='https://198.61.239.128', port=443)

    connect with tls authentication, by providing a hostname, port, a private
    key file (.pem) and certificate (.pem) file
    >>> conn = driver(host='https://198.61.239.128',
    >>> port=4243, key_file='key.pem', cert_file='cert.pem')
    """
    type = Provider.DOCKER
    name = 'Docker'
    website = 'http://docker.io'
    connectionCls = DockerConnection
    supports_clusters = False
    version = '1.24'

    def __init__(self, key='', secret='', secure=False, host='localhost', port=4243, key_file=None, cert_file=None):
        """
        :param    key: API key or username to used (required)
        :type     key: ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :param    key_file: Path to private key for TLS connection (optional)
        :type     key_file: ``str``

        :param    cert_file: Path to public key for TLS connection (optional)
        :type     cert_file: ``str``

        :return: ``None``
        """
        if key_file:
            self.connectionCls = DockertlsConnection
            self.key_file = key_file
            self.cert_file = cert_file
            secure = True
        if host.startswith('https://'):
            secure = True
        prefixes = ['http://', 'https://']
        for prefix in prefixes:
            if host.startswith(prefix):
                host = host.strip(prefix)
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, key_file=key_file, cert_file=cert_file)
        if key_file or cert_file:
            if not (key_file and cert_file):
                raise Exception('Needs both private key file and certificate file for tls authentication')
        self.connection.secure = secure
        self.connection.host = host
        self.connection.port = port
        self.version = self._get_api_version()

    def _ex_connection_class_kwargs(self):
        kwargs = {}
        if hasattr(self, 'key_file'):
            kwargs['key_file'] = self.key_file
        if hasattr(self, 'cert_file'):
            kwargs['cert_file'] = self.cert_file
        return kwargs

    def install_image(self, path):
        """
        Install a container image from a remote path.

        :param path: Path to the container image
        :type  path: ``str``

        :rtype: :class:`libcloud.container.base.ContainerImage`
        """
        payload = {}
        data = json.dumps(payload)
        result = self.connection.request('/v{}/images/create?fromImage={}'.format(self.version, path), data=data, method='POST')
        if 'errorDetail' in result.body:
            raise DockerException(None, result.body)
        image_id = None
        if re.search('Downloaded newer image', result.body) or re.search('"Status: Image is up to date', result.body):
            if re.search('sha256:(?P<id>[a-z0-9]{64})', result.body):
                image_id = re.findall('sha256:(?P<id>[a-z0-9]{64})', result.body)[-1]
        if image_id is None:
            raise DockerException(None, 'failed to install image')
        image = ContainerImage(id=image_id, name=path, path=path, version=None, driver=self.connection.driver, extra={})
        return image

    def list_images(self):
        """
        List the installed container images

        :rtype: ``list`` of :class:`libcloud.container.base.ContainerImage`
        """
        result = self.connection.request('/v%s/images/json' % self.version).object
        images = []
        for image in result:
            try:
                name = image.get('RepoTags')[0]
            except Exception:
                name = image.get('Id')
            images.append(ContainerImage(id=image.get('Id'), name=name, path=name, version=None, driver=self.connection.driver, extra={'created': image.get('Created'), 'size': image.get('Size'), 'virtual_size': image.get('VirtualSize')}))
        return images

    def list_containers(self, image=None, all=True):
        """
        List the deployed container images

        :param image: Filter to containers with a certain image
        :type  image: :class:`libcloud.container.base.ContainerImage`

        :param all: Show all container (including stopped ones)
        :type  all: ``bool``

        :rtype: ``list`` of :class:`libcloud.container.base.Container`
        """
        if all:
            ex = '?all=1'
        else:
            ex = ''
        try:
            result = self.connection.request('/v{}/containers/json{}'.format(self.version, ex)).object
        except Exception as exc:
            errno = getattr(exc, 'errno', None)
            if errno == 111:
                raise DockerException(errno, 'Make sure docker host is accessibleand the API port is correct')
            raise
        containers = [self._to_container(value) for value in result]
        return containers

    def deploy_container(self, name, image, parameters=None, start=True, command=None, hostname=None, user='', stdin_open=True, tty=True, mem_limit=0, ports=None, environment=None, dns=None, volumes=None, volumes_from=None, network_disabled=False, entrypoint=None, cpu_shares=None, working_dir='', domainname=None, memswap_limit=0, port_bindings=None, network_mode='bridge', labels=None):
        """
        Deploy an installed container image

        For details on the additional parameters see : http://bit.ly/1PjMVKV

        :param name: The name of the new container
        :type  name: ``str``

        :param image: The container image to deploy
        :type  image: :class:`libcloud.container.base.ContainerImage`

        :param parameters: Container Image parameters
        :type  parameters: ``str``

        :param start: Start the container on deployment
        :type  start: ``bool``

        :rtype: :class:`Container`
        """
        command = shlex.split(str(command))
        if port_bindings is None:
            port_bindings = {}
        params = {'name': name}
        payload = {'Hostname': hostname, 'Domainname': domainname, 'ExposedPorts': ports, 'User': user, 'Tty': tty, 'OpenStdin': stdin_open, 'StdinOnce': False, 'Memory': mem_limit, 'AttachStdin': True, 'AttachStdout': True, 'AttachStderr': True, 'Env': environment, 'Cmd': command, 'Dns': dns, 'Image': image.name, 'Volumes': volumes, 'VolumesFrom': volumes_from, 'NetworkDisabled': network_disabled, 'Entrypoint': entrypoint, 'CpuShares': cpu_shares, 'WorkingDir': working_dir, 'MemorySwap': memswap_limit, 'PublishAllPorts': True, 'PortBindings': port_bindings, 'NetworkMode': network_mode, 'Labels': labels}
        data = json.dumps(payload)
        try:
            result = self.connection.request('/v%s/containers/create' % self.version, data=data, params=params, method='POST')
        except Exception as e:
            message = e.message or str(e)
            if message.startswith('No such image:'):
                raise DockerException(None, 'No such image: %s' % image.name)
            else:
                raise DockerException(None, e)
        id_ = result.object['Id']
        payload = {'Binds': [], 'PublishAllPorts': True, 'PortBindings': port_bindings}
        data = json.dumps(payload)
        if start:
            if float(self._get_api_version()) > 1.22:
                result = self.connection.request('/v{}/containers/{}/start'.format(self.version, id_), method='POST')
            else:
                result = self.connection.request('/v{}/containers/{}/start'.format(self.version, id_), data=data, method='POST')
        return self.get_container(id_)

    def get_container(self, id):
        """
        Get a container by ID

        :param id: The ID of the container to get
        :type  id: ``str``

        :rtype: :class:`libcloud.container.base.Container`
        """
        result = self.connection.request('/v{}/containers/{}/json'.format(self.version, id)).object
        return self._to_container(result)

    def start_container(self, container):
        """
        Start a container

        :param container: The container to be started
        :type  container: :class:`libcloud.container.base.Container`

        :return: The container refreshed with current data
        :rtype: :class:`libcloud.container.base.Container`
        """
        if float(self._get_api_version()) > 1.22:
            result = self.connection.request('/v{}/containers/{}/start'.format(self.version, container.id), method='POST')
        else:
            payload = {'Binds': [], 'PublishAllPorts': True}
            data = json.dumps(payload)
            result = self.connection.request('/v{}/containers/{}/start'.format(self.version, container.id), method='POST', data=data)
        if result.status in VALID_RESPONSE_CODES:
            return self.get_container(container.id)
        else:
            raise DockerException(result.status, 'failed to start container')

    def stop_container(self, container):
        """
        Stop a container

        :param container: The container to be stopped
        :type  container: :class:`libcloud.container.base.Container`

        :return: The container refreshed with current data
        :rtype: :class:`libcloud.container.base.Container`
        """
        result = self.connection.request('/v{}/containers/{}/stop'.format(self.version, container.id), method='POST')
        if result.status in VALID_RESPONSE_CODES:
            return self.get_container(container.id)
        else:
            raise DockerException(result.status, 'failed to stop container')

    def restart_container(self, container):
        """
        Restart a container

        :param container: The container to be stopped
        :type  container: :class:`libcloud.container.base.Container`

        :return: The container refreshed with current data
        :rtype: :class:`libcloud.container.base.Container`
        """
        data = json.dumps({'t': 10})
        result = self.connection.request('/v{}/containers/{}/restart'.format(self.version, container.id), data=data, method='POST')
        if result.status in VALID_RESPONSE_CODES:
            return self.get_container(container.id)
        else:
            raise DockerException(result.status, 'failed to restart container')

    def destroy_container(self, container):
        """
        Remove a container

        :param container: The container to be destroyed
        :type  container: :class:`libcloud.container.base.Container`

        :return: True if the destroy was successful, False otherwise.
        :rtype: ``bool``
        """
        result = self.connection.request('/v{}/containers/{}'.format(self.version, container.id), method='DELETE')
        return result.status in VALID_RESPONSE_CODES

    def ex_list_processes(self, container):
        """
        List processes running inside a container

        :param container: The container to list processes for.
        :type  container: :class:`libcloud.container.base.Container`

        :rtype: ``str``
        """
        result = self.connection.request('/v{}/containers/{}/top'.format(self.version, container.id)).object
        return result

    def ex_rename_container(self, container, name):
        """
        Rename a container

        :param container: The container to be renamed
        :type  container: :class:`libcloud.container.base.Container`

        :param name: The new name
        :type  name: ``str``

        :rtype: :class:`libcloud.container.base.Container`
        """
        result = self.connection.request('/v{}/containers/{}/rename?name={}'.format(self.version, container.id, name), method='POST')
        if result.status in VALID_RESPONSE_CODES:
            return self.get_container(container.id)

    def ex_get_logs(self, container, stream=False):
        """
        Get container logs

        If stream == True, logs will be yielded as a stream
        From Api Version 1.11 and above we need a GET request to get the logs
        Logs are in different format of those of Version 1.10 and below

        :param container: The container to list logs for
        :type  container: :class:`libcloud.container.base.Container`

        :param stream: Stream the output
        :type  stream: ``bool``

        :rtype: ``bool``
        """
        payload = {}
        data = json.dumps(payload)
        if float(self._get_api_version()) > 1.1:
            result = self.connection.request('/v%s/containers/%s/logs?follow=%s&stdout=1&stderr=1' % (self.version, container.id, str(stream))).object
            logs = result
        else:
            result = self.connection.request('/v%s/containers/%s/attach?logs=1&stream=%s&stdout=1&stderr=1' % (self.version, container.id, str(stream)), method='POST', data=data)
            logs = result.body
        return logs

    def ex_search_images(self, term):
        """Search for an image on Docker.io.
        Returns a list of ContainerImage objects

        >>> images = conn.ex_search_images(term='mistio')
        >>> images
        [<ContainerImage: id=rolikeusch/docker-mistio...>,
         <ContainerImage: id=mist/mistio, name=mist/mistio,
             driver=Docker  ...>]

         :param term: The search term
         :type  term: ``str``

         :rtype: ``list`` of :class:`libcloud.container.base.ContainerImage`
        """
        term = term.replace(' ', '+')
        result = self.connection.request('/v{}/images/search?term={}'.format(self.version, term)).object
        images = []
        for image in result:
            name = image.get('name')
            images.append(ContainerImage(id=name, path=name, version=None, name=name, driver=self.connection.driver, extra={'description': image.get('description'), 'is_official': image.get('is_official'), 'is_trusted': image.get('is_trusted'), 'star_count': image.get('star_count')}))
        return images

    def ex_delete_image(self, image):
        """
        Remove image from the filesystem

        :param  image: The image to remove
        :type   image: :class:`libcloud.container.base.ContainerImage`

        :rtype: ``bool``
        """
        result = self.connection.request('/v{}/images/{}'.format(self.version, image.name), method='DELETE')
        return result.status in VALID_RESPONSE_CODES

    def _to_container(self, data):
        """
        Convert container in Container instances
        """
        try:
            name = data.get('Name').strip('/')
        except Exception:
            try:
                name = data.get('Names')[0].strip('/')
            except Exception:
                name = data.get('Id')
        state = data.get('State')
        if isinstance(state, dict):
            status = data.get('Status', state.get('Status') if state is not None else None)
        else:
            status = data.get('Status')
        if 'Exited' in status:
            state = ContainerState.STOPPED
        elif status.startswith('Up '):
            state = ContainerState.RUNNING
        elif 'running' in status:
            state = ContainerState.RUNNING
        else:
            state = ContainerState.STOPPED
        image = data.get('Image')
        ports = data.get('Ports', [])
        created = data.get('Created')
        if isinstance(created, float):
            created = ts_to_str(created)
        extra = {'id': data.get('Id'), 'status': data.get('Status'), 'created': created, 'image': image, 'ports': ports, 'command': data.get('Command'), 'sizerw': data.get('SizeRw'), 'sizerootfs': data.get('SizeRootFs')}
        ips = []
        if ports is not None:
            for port in ports:
                if port.get('IP') is not None:
                    ips.append(port.get('IP'))
        return Container(id=data['Id'], name=name, image=ContainerImage(id=data.get('ImageID', None), path=image, name=image, version=None, driver=self.connection.driver), ip_addresses=ips, state=state, driver=self.connection.driver, extra=extra)

    def _get_api_version(self):
        """
        Get the docker API version information
        """
        result = self.connection.request('/version').object
        result = result or {}
        api_version = result.get('ApiVersion')
        return api_version