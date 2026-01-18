import os
import re
import time
import atexit
import random
import socket
import hashlib
import binascii
import datetime
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Type, Tuple, Union, Callable, Optional
import libcloud.compute.ssh
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import b
from libcloud.common.base import BaseDriver, Connection, ConnectionKey
from libcloud.compute.ssh import SSHClient, BaseSSHClient, SSHCommandTimeoutError, have_paramiko
from libcloud.common.types import LibcloudError
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet, is_valid_ip_address
class NodeDriver(BaseDriver):
    """
    A base NodeDriver class to derive from

    This class is always subclassed by a specific driver.  For
    examples of base behavior of most functions (except deploy node)
    see the dummy driver.
    """
    connectionCls = ConnectionKey
    name = None
    api_name = None
    website = None
    type = None
    port = None
    features = {'create_node': []}
    '\n    List of available features for a driver.\n        - :meth:`libcloud.compute.base.NodeDriver.create_node`\n            - ssh_key: Supports :class:`.NodeAuthSSHKey` as an authentication\n              method for nodes.\n            - password: Supports :class:`.NodeAuthPassword` as an\n              authentication\n              method for nodes.\n            - generates_password: Returns a password attribute on the Node\n              object returned from creation.\n    '
    NODE_STATE_MAP = {}

    def list_nodes(self, *args, **kwargs):
        """
        List all nodes.

        :return:  list of node objects
        :rtype: ``list`` of :class:`.Node`
        """
        raise NotImplementedError('list_nodes not implemented for this driver')

    def list_sizes(self, location=None):
        """
        List sizes on a provider

        :param location: The location at which to list sizes
        :type location: :class:`.NodeLocation`

        :return: list of node size objects
        :rtype: ``list`` of :class:`.NodeSize`
        """
        raise NotImplementedError('list_sizes not implemented for this driver')

    def list_locations(self):
        """
        List data centers for a provider

        :return: list of node location objects
        :rtype: ``list`` of :class:`.NodeLocation`
        """
        raise NotImplementedError('list_locations not implemented for this driver')

    def create_node(self, name, size, image, location=None, auth=None):
        """
        Create a new node instance. This instance will be started
        automatically.

        Not all hosting API's are created equal and to allow libcloud to
        support as many as possible there are some standard supported
        variations of ``create_node``. These are declared using a
        ``features`` API.
        You can inspect ``driver.features['create_node']`` to see what
        variation of the API you are dealing with:

        ``ssh_key``
            You can inject a public key into a new node allows key based SSH
            authentication.
        ``password``
            You can inject a password into a new node for SSH authentication.
            If no password is provided libcloud will generated a password.
            The password will be available as
            ``return_value.extra['password']``.
        ``generates_password``
            The hosting provider will generate a password. It will be returned
            to you via ``return_value.extra['password']``.

        Some drivers allow you to set how you will authenticate with the
        instance that is created. You can inject this initial authentication
        information via the ``auth`` parameter.

        If a driver supports the ``ssh_key`` feature flag for ``created_node``
        you can upload a public key into the new instance::

            >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
            >>> driver = DummyNodeDriver(0)
            >>> auth = NodeAuthSSHKey('pubkey data here')
            >>> node = driver.create_node("test_node", auth=auth)

        If a driver supports the ``password`` feature flag for ``create_node``
        you can set a password::

            >>> driver = DummyNodeDriver(0)
            >>> auth = NodeAuthPassword('mysecretpassword')
            >>> node = driver.create_node("test_node", auth=auth)

        If a driver supports the ``password`` feature and you don't provide the
        ``auth`` argument libcloud will assign a password::

            >>> driver = DummyNodeDriver(0)
            >>> node = driver.create_node("test_node")
            >>> password = node.extra['password']

        A password will also be returned in this way for drivers that declare
        the ``generates_password`` feature, though in that case the password is
        actually provided to the driver API by the hosting provider rather than
        generated by libcloud.

        You can only pass a :class:`.NodeAuthPassword` or
        :class:`.NodeAuthSSHKey` to ``create_node`` via the auth parameter if
        has the corresponding feature flag.

        :param name:   String with a name for this new node (required)
        :type name:   ``str``

        :param size:   The size of resources allocated to this node.
                            (required)
        :type size:   :class:`.NodeSize`

        :param image:  OS Image to boot on node. (required)
        :type image:  :class:`.NodeImage`

        :param location: Which data center to create a node in. If empty,
                              undefined behavior will be selected. (optional)
        :type location: :class:`.NodeLocation`

        :param auth:   Initial authentication information for the node
                            (optional)
        :type auth:   :class:`.NodeAuthSSHKey` or :class:`NodeAuthPassword`

        :return: The newly created node.
        :rtype: :class:`.Node`
        """
        raise NotImplementedError('create_node not implemented for this driver')

    def deploy_node(self, deploy, ssh_username='root', ssh_alternate_usernames=None, ssh_port=22, ssh_timeout=10, ssh_key=None, ssh_key_password=None, auth=None, timeout=SSH_CONNECT_TIMEOUT, max_tries=3, ssh_interface='public_ips', at_exit_func=None, wait_period=5, **create_node_kwargs):
        """
        Create a new node, and start deployment.

        In order to be able to SSH into a created node access credentials are
        required.

        A user can pass either a :class:`.NodeAuthPassword` or
        :class:`.NodeAuthSSHKey` to the ``auth`` argument. If the
        ``create_node`` implementation supports that kind if credential (as
        declared in ``self.features['create_node']``) then it is passed on to
        ``create_node``. Otherwise it is not passed on to ``create_node`` and
        it is only used for authentication.

        If the ``auth`` parameter is not supplied but the driver declares it
        supports ``generates_password`` then the password returned by
        ``create_node`` will be used to SSH into the server.

        Finally, if the ``ssh_key_file`` is supplied that key will be used to
        SSH into the server.

        This function may raise a :class:`DeploymentException`, if a
        create_node call was successful, but there is a later error (like SSH
        failing or timing out).  This exception includes a Node object which
        you may want to destroy if incomplete deployments are not desirable.

        >>> from libcloud.compute.drivers.dummy import DummyNodeDriver
        >>> from libcloud.compute.deployment import ScriptDeployment
        >>> from libcloud.compute.deployment import MultiStepDeployment
        >>> from libcloud.compute.base import NodeAuthSSHKey
        >>> driver = DummyNodeDriver(0)
        >>> key = NodeAuthSSHKey('...') # read from file
        >>> script = ScriptDeployment("yum -y install emacs strace tcpdump")
        >>> msd = MultiStepDeployment([key, script])
        >>> def d():
        ...     try:
        ...         driver.deploy_node(deploy=msd)
        ...     except NotImplementedError:
        ...         print ("not implemented for dummy driver")
        >>> d()
        not implemented for dummy driver

        Deploy node is typically not overridden in subclasses.  The
        existing implementation should be able to handle most such.

        :param deploy: Deployment to run once machine is online and
                            available to SSH.
        :type deploy: :class:`Deployment`

        :param ssh_username: Optional name of the account which is used
                                  when connecting to
                                  SSH server (default is root)
        :type ssh_username: ``str``

        :param ssh_alternate_usernames: Optional list of ssh usernames to
                                             try to connect with if using the
                                             default one fails
        :type ssh_alternate_usernames: ``list``

        :param ssh_port: Optional SSH server port (default is 22)
        :type ssh_port: ``int``

        :param ssh_timeout: Optional SSH connection timeout in seconds
                                 (default is 10)
        :type ssh_timeout: ``float``

        :param auth:   Initial authentication information for the node
                            (optional)
        :type auth:   :class:`.NodeAuthSSHKey` or :class:`NodeAuthPassword`

        :param ssh_key: A path (or paths) to an SSH private key with which
                             to attempt to authenticate. (optional)
        :type ssh_key: ``str`` or ``list`` of ``str``

        :param ssh_key_password: Optional password used for encrypted keys.
        :type ssh_key_password: ``str``

        :param timeout: How many seconds to wait before timing out.
                             (default is 600)
        :type timeout: ``int``

        :param max_tries: How many times to retry if a deployment fails
                               before giving up (default is 3)
        :type max_tries: ``int``

        :param ssh_interface: The interface to wait for. Default is
                                   'public_ips', other option is 'private_ips'.
        :type ssh_interface: ``str``

        :param at_exit_func: Optional atexit handler function which will be
                             registered and called with created node if user
                             cancels the deploy process (e.g. CTRL+C), after
                             the node has been created, but before the deploy
                             process has finished.

                             This method gets passed in two keyword arguments:

                             - driver -> node driver in question
                             - node -> created Node object

                             Keep in mind that this function will only be
                             called in such scenario. In case the method
                             finishes (this includes throwing an exception),
                             at exit handler function won't be called.
        :type at_exit_func: ``func``

        :param wait_period: How many seconds to wait between each iteration
                            while waiting for node to transition into
                            running state and have IP assigned. (default is 5)
        :type wait_period: ``int``

        """
        if not libcloud.compute.ssh.have_paramiko:
            raise RuntimeError('paramiko is not installed. You can install ' + 'it using pip: pip install paramiko')
        if auth:
            if not isinstance(auth, (NodeAuthSSHKey, NodeAuthPassword)):
                raise NotImplementedError('If providing auth, only NodeAuthSSHKey orNodeAuthPassword is supported')
        elif ssh_key:
            pass
        elif 'create_node' in self.features:
            f = self.features['create_node']
            if 'generates_password' not in f and 'password' not in f:
                raise NotImplementedError('deploy_node not implemented for this driver')
        else:
            raise NotImplementedError('deploy_node not implemented for this driver')
        try:
            if auth:
                node = self.create_node(auth=auth, **create_node_kwargs)
            else:
                node = self.create_node(**create_node_kwargs)
        except TypeError as e:
            msg_1_re = 'create_node\\(\\) missing \\d+ required positional arguments.*'
            msg_2_re = 'create_node\\(\\) takes at least \\d+ arguments.*'
            if re.match(msg_1_re, str(e)) or re.match(msg_2_re, str(e)):
                node = self.create_node(deploy=deploy, ssh_username=ssh_username, ssh_alternate_usernames=ssh_alternate_usernames, ssh_port=ssh_port, ssh_timeout=ssh_timeout, ssh_key=ssh_key, auth=auth, timeout=timeout, max_tries=max_tries, ssh_interface=ssh_interface, **create_node_kwargs)
            else:
                raise e
        if at_exit_func:
            atexit.register(at_exit_func, driver=self, node=node)
        password = None
        if auth:
            if isinstance(auth, NodeAuthPassword):
                password = auth.password
        elif 'password' in node.extra:
            password = node.extra['password']
        wait_timeout = timeout or NODE_ONLINE_WAIT_TIMEOUT
        try:
            node, ip_addresses = self.wait_until_running(nodes=[node], wait_period=wait_period, timeout=wait_timeout, ssh_interface=ssh_interface)[0]
        except Exception as e:
            if at_exit_func:
                atexit.unregister(at_exit_func)
            raise DeploymentError(node=node, original_exception=e, driver=self)
        ssh_alternate_usernames = ssh_alternate_usernames or []
        deploy_timeout = timeout or SSH_CONNECT_TIMEOUT
        deploy_error = None
        for username in [ssh_username] + ssh_alternate_usernames:
            try:
                self._connect_and_run_deployment_script(task=deploy, node=node, ssh_hostname=ip_addresses[0], ssh_port=ssh_port, ssh_username=username, ssh_password=password, ssh_key_file=ssh_key, ssh_key_password=ssh_key_password, ssh_timeout=ssh_timeout, timeout=deploy_timeout, max_tries=max_tries)
            except Exception as e:
                deploy_error = e
            else:
                deploy_error = None
                break
        if deploy_error is not None:
            if at_exit_func:
                atexit.unregister(at_exit_func)
            raise DeploymentError(node=node, original_exception=deploy_error, driver=self)
        if at_exit_func:
            atexit.unregister(at_exit_func)
        return node

    def reboot_node(self, node):
        """
        Reboot a node.

        :param node: The node to be rebooted
        :type node: :class:`.Node`

        :return: True if the reboot was successful, otherwise False
        :rtype: ``bool``
        """
        raise NotImplementedError('reboot_node not implemented for this driver')

    def start_node(self, node):
        """
        Start a node.

        :param node: The node to be started
        :type node: :class:`.Node`

        :return: True if the start was successful, otherwise False
        :rtype: ``bool``
        """
        raise NotImplementedError('start_node not implemented for this driver')

    def stop_node(self, node):
        """
        Stop a node

        :param node: The node to be stopped.
        :type node: :class:`.Node`

        :return: True if the stop was successful, otherwise False
        :rtype: ``bool``
        """
        raise NotImplementedError('stop_node not implemented for this driver')

    def destroy_node(self, node):
        """
        Destroy a node.

        Depending upon the provider, this may destroy all data associated with
        the node, including backups.

        :param node: The node to be destroyed
        :type node: :class:`.Node`

        :return: True if the destroy was successful, False otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('destroy_node not implemented for this driver')

    def list_volumes(self):
        """
        List storage volumes.

        :rtype: ``list`` of :class:`.StorageVolume`
        """
        raise NotImplementedError('list_volumes not implemented for this driver')

    def list_volume_snapshots(self, volume):
        """
        List snapshots for a storage volume.

        :rtype: ``list`` of :class:`VolumeSnapshot`
        """
        raise NotImplementedError('list_volume_snapshots not implemented for this driver')

    def create_volume(self, size, name, location=None, snapshot=None):
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
        :type snapshot: :class:`.VolumeSnapshot`

        :return: The newly created volume.
        :rtype: :class:`StorageVolume`
        """
        raise NotImplementedError('create_volume not implemented for this driver')

    def create_volume_snapshot(self, volume, name=None):
        """
        Creates a snapshot of the storage volume.

        :param volume: The StorageVolume to create a VolumeSnapshot from
        :type volume: :class:`.StorageVolume`

        :param name: Name of created snapshot (optional)
        :type name: `str`

        :rtype: :class:`VolumeSnapshot`
        """
        raise NotImplementedError('create_volume_snapshot not implemented for this driver')

    def attach_volume(self, node, volume, device=None):
        """
        Attaches volume to node.

        :param node: Node to attach volume to.
        :type node: :class:`.Node`

        :param volume: Volume to attach.
        :type volume: :class:`.StorageVolume`

        :param device: Where the device is exposed, e.g. '/dev/sdb'
        :type device: ``str``

        :rytpe: ``bool``
        """
        raise NotImplementedError('attach not implemented for this driver')

    def detach_volume(self, volume):
        """
        Detaches a volume from a node.

        :param volume: Volume to be detached
        :type volume: :class:`.StorageVolume`

        :rtype: ``bool``
        """
        raise NotImplementedError('detach not implemented for this driver')

    def destroy_volume(self, volume):
        """
        Destroys a storage volume.

        :param volume: Volume to be destroyed
        :type volume: :class:`StorageVolume`

        :rtype: ``bool``
        """
        raise NotImplementedError('destroy_volume not implemented for this driver')

    def destroy_volume_snapshot(self, snapshot):
        """
        Destroys a snapshot.

        :param snapshot: The snapshot to delete
        :type snapshot: :class:`VolumeSnapshot`

        :rtype: :class:`bool`
        """
        raise NotImplementedError('destroy_volume_snapshot not implemented for this driver')

    def list_images(self, location=None):
        """
        List images on a provider.

        :param location: The location at which to list images.
        :type location: :class:`.NodeLocation`

        :return: list of node image objects.
        :rtype: ``list`` of :class:`.NodeImage`
        """
        raise NotImplementedError('list_images not implemented for this driver')

    def create_image(self, node, name, description=None):
        """
        Creates an image from a node object.

        :param node: Node to run the task on.
        :type node: :class:`.Node`

        :param name: name for new image.
        :type name: ``str``

        :param description: description for new image.
        :type name: ``description``

        :rtype: :class:`.NodeImage`:
        :return: NodeImage instance on success.

        """
        raise NotImplementedError('create_image not implemented for this driver')

    def delete_image(self, node_image):
        """
        Deletes a node image from a provider.

        :param node_image: Node image object.
        :type node_image: :class:`.NodeImage`

        :return: ``True`` if delete_image was successful, ``False`` otherwise.
        :rtype: ``bool``
        """
        raise NotImplementedError('delete_image not implemented for this driver')

    def get_image(self, image_id):
        """
        Returns a single node image from a provider.

        :param image_id: Node to run the task on.
        :type image_id: ``str``

        :rtype :class:`.NodeImage`:
        :return: NodeImage instance on success.
        """
        raise NotImplementedError('get_image not implemented for this driver')

    def copy_image(self, source_region, node_image, name, description=None):
        """
        Copies an image from a source region to the current region.

        :param source_region: Region to copy the node from.
        :type source_region: ``str``

        :param node_image: NodeImage to copy.
        :type node_image: :class:`.NodeImage`:

        :param name: name for new image.
        :type name: ``str``

        :param description: description for new image.
        :type name: ``str``

        :rtype: :class:`.NodeImage`:
        :return: NodeImage instance on success.
        """
        raise NotImplementedError('copy_image not implemented for this driver')

    def list_key_pairs(self):
        """
        List all the available key pair objects.

        :rtype: ``list`` of :class:`.KeyPair` objects
        """
        raise NotImplementedError('list_key_pairs not implemented for this driver')

    def get_key_pair(self, name):
        """
        Retrieve a single key pair.

        :param name: Name of the key pair to retrieve.
        :type name: ``str``

        :rtype: :class:`.KeyPair`
        """
        raise NotImplementedError('get_key_pair not implemented for this driver')

    def create_key_pair(self, name):
        """
        Create a new key pair object.

        :param name: Key pair name.
        :type name: ``str``

        :rtype: :class:`.KeyPair` object
        """
        raise NotImplementedError('create_key_pair not implemented for this driver')

    def import_key_pair_from_string(self, name, key_material):
        """
        Import a new public key from string.

        :param name: Key pair name.
        :type name: ``str``

        :param key_material: Public key material.
        :type key_material: ``str``

        :rtype: :class:`.KeyPair` object
        """
        raise NotImplementedError('import_key_pair_from_string not implemented for this driver')

    def import_key_pair_from_file(self, name, key_file_path):
        """
        Import a new public key from string.

        :param name: Key pair name.
        :type name: ``str``

        :param key_file_path: Path to the public key file.
        :type key_file_path: ``str``

        :rtype: :class:`.KeyPair` object
        """
        key_file_path = os.path.expanduser(key_file_path)
        with open(key_file_path) as fp:
            key_material = fp.read().strip()
        return self.import_key_pair_from_string(name=name, key_material=key_material)

    def delete_key_pair(self, key_pair):
        """
        Delete an existing key pair.

        :param key_pair: Key pair object.
        :type key_pair: :class:`.KeyPair`

        :rtype: ``bool``
        """
        raise NotImplementedError('delete_key_pair not implemented for this driver')

    def wait_until_running(self, nodes, wait_period=5, timeout=600, ssh_interface='public_ips', force_ipv4=True, ex_list_nodes_kwargs=None):
        """
        Block until the provided nodes are considered running.

        Node is considered running when it's state is "running" and when it has
        at least one IP address assigned.

        :param nodes: List of nodes to wait for.
        :type nodes: ``list`` of :class:`.Node`

        :param wait_period: How many seconds to wait between each loop
                            iteration. (default is 3)
        :type wait_period: ``int``

        :param timeout: How many seconds to wait before giving up.
                        (default is 600)
        :type timeout: ``int``

        :param ssh_interface: Which attribute on the node to use to obtain
                              an IP address. Valid options: public_ips,
                              private_ips. Default is public_ips.
        :type ssh_interface: ``str``

        :param force_ipv4: Ignore IPv6 addresses (default is True).
        :type force_ipv4: ``bool``

        :param ex_list_nodes_kwargs: Optional driver-specific keyword arguments
                                     which are passed to the ``list_nodes``
                                     method.
        :type ex_list_nodes_kwargs: ``dict``

        :return: ``[(Node, ip_addresses)]`` list of tuple of Node instance and
                 list of ip_address on success.
        :rtype: ``list`` of ``tuple``
        """
        ex_list_nodes_kwargs = ex_list_nodes_kwargs or {}

        def is_supported(address):
            """
            Return True for supported address.
            """
            if force_ipv4 and (not is_valid_ip_address(address=address, family=socket.AF_INET)):
                return False
            return True

        def filter_addresses(addresses):
            """
            Return list of supported addresses.
            """
            return [address for address in addresses if is_supported(address)]
        if ssh_interface not in ['public_ips', 'private_ips']:
            raise ValueError('ssh_interface argument must either be ' + 'public_ips or private_ips')
        start = time.time()
        end = start + timeout
        uuids = {node.uuid for node in nodes}
        while time.time() < end:
            all_nodes = self.list_nodes(**ex_list_nodes_kwargs)
            matching_nodes = list((node for node in all_nodes if node.uuid in uuids))
            if len(matching_nodes) > len(uuids):
                found_uuids = [node.uuid for node in matching_nodes]
                msg = 'Unable to match specified uuids ' + '(%s) with existing nodes. Found ' % uuids + 'multiple nodes with same uuid: (%s)' % found_uuids
                raise LibcloudError(value=msg, driver=self)
            running_nodes = [node for node in matching_nodes if node.state == NodeState.RUNNING]
            addresses = []
            for node in running_nodes:
                node_addresses = filter_addresses(getattr(node, ssh_interface))
                if len(node_addresses) >= 1:
                    addresses.append(node_addresses)
            if len(running_nodes) == len(uuids) == len(addresses):
                return list(zip(running_nodes, addresses))
            else:
                time.sleep(wait_period)
                continue
        raise LibcloudError(value='Timed out after %s seconds' % timeout, driver=self)

    def _get_and_check_auth(self, auth):
        """
        Helper function for providers supporting :class:`.NodeAuthPassword` or
        :class:`.NodeAuthSSHKey`

        Validates that only a supported object type is passed to the auth
        parameter and raises an exception if it is not.

        If no :class:`.NodeAuthPassword` object is provided but one is expected
        then a password is automatically generated.
        """
        if isinstance(auth, NodeAuthPassword):
            if 'password' in self.features['create_node']:
                return auth
            raise LibcloudError('Password provided as authentication information, but passwordnot supported', driver=self)
        if isinstance(auth, NodeAuthSSHKey):
            if 'ssh_key' in self.features['create_node']:
                return auth
            raise LibcloudError('SSH Key provided as authentication information, but SSH Keynot supported', driver=self)
        if 'password' in self.features['create_node']:
            value = os.urandom(16)
            value = binascii.hexlify(value).decode('ascii')
            password = ''
            for char in value:
                if not char.isdigit() and char.islower():
                    if random.randint(0, 1) == 1:
                        char = char.upper()
                password += char
            return NodeAuthPassword(password, generated=True)
        if auth:
            raise LibcloudError('"auth" argument provided, but it was not a NodeAuthPasswordor NodeAuthSSHKey object', driver=self)

    def _wait_until_running(self, node, wait_period=3, timeout=600, ssh_interface='public_ips', force_ipv4=True):
        return self.wait_until_running(nodes=[node], wait_period=wait_period, timeout=timeout, ssh_interface=ssh_interface, force_ipv4=force_ipv4)

    def _ssh_client_connect(self, ssh_client, wait_period=1.5, timeout=300):
        """
        Try to connect to the remote SSH server. If a connection times out or
        is refused it is retried up to timeout number of seconds.

        :param ssh_client: A configured SSHClient instance
        :type ssh_client: ``SSHClient``

        :param wait_period: How many seconds to wait between each loop
                            iteration. (default is 1.5)
        :type wait_period: ``int``

        :param timeout: How many seconds to wait before giving up.
                        (default is 300)
        :type timeout: ``int``

        :return: ``SSHClient`` on success
        """
        start = time.time()
        end = start + timeout
        while time.time() < end:
            try:
                ssh_client.connect()
            except SSH_TIMEOUT_EXCEPTION_CLASSES as e:
                message = str(e).lower()
                for fatal_msg in SSH_FATAL_ERROR_MSGS:
                    if fatal_msg in message:
                        raise e
                try:
                    ssh_client.close()
                except Exception:
                    pass
                time.sleep(wait_period)
                continue
            else:
                return ssh_client
        raise LibcloudError(value='Could not connect to the remote SSH ' + 'server. Giving up.', driver=self)

    def _connect_and_run_deployment_script(self, task, node, ssh_hostname, ssh_port, ssh_username, ssh_password, ssh_key_file, ssh_key_password, ssh_timeout, timeout, max_tries):
        """
        Establish an SSH connection to the node and run the provided deployment
        task.

        :rtype: :class:`.Node`:
        :return: Node instance on success.
        """
        ssh_client = SSHClient(hostname=ssh_hostname, port=ssh_port, username=ssh_username, password=ssh_key_password or ssh_password, key_files=ssh_key_file, timeout=ssh_timeout)
        ssh_client = self._ssh_client_connect(ssh_client=ssh_client, timeout=timeout)
        node = self._run_deployment_script(task=task, node=node, ssh_client=ssh_client, max_tries=max_tries)
        return node

    def _run_deployment_script(self, task, node, ssh_client, max_tries=3):
        """
        Run the deployment script on the provided node. At this point it is
        assumed that SSH connection has already been established.

        :param task: Deployment task to run.
        :type task: :class:`Deployment`

        :param node: Node to run the task on.
        :type node: ``Node``

        :param ssh_client: A configured and connected SSHClient instance.
        :type ssh_client: :class:`SSHClient`

        :param max_tries: How many times to retry if a deployment fails
                          before giving up. (default is 3)
        :type max_tries: ``int``

        :rtype: :class:`.Node`
        :return: ``Node`` Node instance on success.
        """
        tries = 0
        while tries < max_tries:
            try:
                node = task.run(node, ssh_client)
            except SSHCommandTimeoutError as e:
                raise e
            except Exception as e:
                tries += 1
                if 'ssh session not active' in str(e).lower():
                    try:
                        ssh_client.close()
                    except Exception:
                        pass
                    timeout = int(ssh_client.timeout) if ssh_client.timeout else 10
                    ssh_client = self._ssh_client_connect(ssh_client=ssh_client, timeout=timeout)
                if tries >= max_tries:
                    tb = traceback.format_exc()
                    raise LibcloudError(value='Failed after %d tries: %s.\n%s' % (max_tries, str(e), tb), driver=self)
            else:
                ssh_client.close()
                return node
        return node

    def _get_size_price(self, size_id):
        """
        Return pricing information for the provided size id.
        """
        return get_size_price(driver_type='compute', driver_name=self.api_name, size_id=size_id)