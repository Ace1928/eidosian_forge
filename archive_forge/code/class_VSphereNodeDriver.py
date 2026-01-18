import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class VSphereNodeDriver(NodeDriver):
    name = 'VMware vSphere'
    website = 'http://www.vmware.com/products/vsphere/'
    type = Provider.VSPHERE
    NODE_STATE_MAP = {'poweredOn': NodeState.RUNNING, 'poweredOff': NodeState.STOPPED, 'suspended': NodeState.SUSPENDED}

    def __init__(self, host, username, password, port=443, ca_cert=None):
        """Initialize a connection by providing a hostname,
        username and password
        """
        if pyvmomi is None:
            raise ImportError('Missing "pyvmomi" dependency. You can install it using pip - pip install pyvmomi')
        self.host = host
        try:
            if ca_cert is None:
                self.connection = connect.SmartConnect(host=host, port=port, user=username, pwd=password)
            else:
                context = ssl.create_default_context(cafile=ca_cert)
                self.connection = connect.SmartConnect(host=host, port=port, user=username, pwd=password, sslContext=context)
            atexit.register(connect.Disconnect, self.connection)
        except Exception as exc:
            error_message = str(exc).lower()
            if 'incorrect user name' in error_message:
                raise InvalidCredsError('Check your username and password are valid')
            if 'connection refused' in error_message or 'is not a vim server' in error_message:
                raise LibcloudError('Check that the host provided is a vSphere installation', driver=self)
            if 'name or service not known' in error_message:
                raise LibcloudError('Check that the vSphere host is accessible', driver=self)
            if 'certificate verify failed' in error_message:
                try:
                    context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
                    context.verify_mode = ssl.CERT_NONE
                except ImportError:
                    raise ImportError('To use self signed certificates, please upgrade to python 2.7.11 and pyvmomi 6.0.0+')
                self.connection = connect.SmartConnect(host=host, port=port, user=username, pwd=password, sslContext=context)
                atexit.register(connect.Disconnect, self.connection)
            else:
                raise LibcloudError('Cannot connect to vSphere', driver=self)

    def list_locations(self, ex_show_hosts_in_drs=True):
        """
        Lists locations
        """
        content = self.connection.RetrieveContent()
        potential_locations = [dc for dc in content.viewManager.CreateContainerView(content.rootFolder, [vim.ClusterComputeResource, vim.HostSystem], recursive=True).view]
        locations = []
        hosts_all = []
        clusters = []
        for location in potential_locations:
            if isinstance(location, vim.HostSystem):
                hosts_all.append(location)
            elif isinstance(location, vim.ClusterComputeResource):
                if location.configuration.drsConfig.enabled:
                    clusters.append(location)
        if ex_show_hosts_in_drs:
            hosts = hosts_all
        else:
            hosts_filter = [host for cluster in clusters for host in cluster.host]
            hosts = [host for host in hosts_all if host not in hosts_filter]
        for cluster in clusters:
            locations.append(self._to_location(cluster))
        for host in hosts:
            locations.append(self._to_location(host))
        return locations

    def _to_location(self, data):
        try:
            if isinstance(data, vim.HostSystem):
                extra = {'type': 'host', 'state': data.runtime.connectionState, 'hypervisor': data.config.product.fullName, 'vendor': data.hardware.systemInfo.vendor, 'model': data.hardware.systemInfo.model, 'ram': data.hardware.memorySize, 'cpu': {'packages': data.hardware.cpuInfo.numCpuPackages, 'cores': data.hardware.cpuInfo.numCpuCores, 'threads': data.hardware.cpuInfo.numCpuThreads}, 'uptime': data.summary.quickStats.uptime, 'parent': str(data.parent)}
            elif isinstance(data, vim.ClusterComputeResource):
                extra = {'type': 'cluster', 'overallStatus': data.overallStatus, 'drs': data.configuration.drsConfig.enabled, 'hosts': [host.name for host in data.host], 'parent': str(data.parent)}
        except AttributeError as exc:
            logger.error('Cannot convert location {}: {!r}'.format(data.name, exc))
            extra = {}
        return NodeLocation(id=data.name, name=data.name, country=None, extra=extra, driver=self)

    def ex_list_networks(self):
        """
        List networks
        """
        content = self.connection.RetrieveContent()
        networks = content.viewManager.CreateContainerView(content.rootFolder, [vim.Network], recursive=True).view
        return [self._to_network(network) for network in networks]

    def _to_network(self, data):
        summary = data.summary
        extra = {'hosts': [h.name for h in data.host], 'ip_pool_name': summary.ipPoolName, 'ip_pool_id': summary.ipPoolId, 'accessible': summary.accessible}
        return VSphereNetwork(id=data.name, name=data.name, extra=extra)

    def list_sizes(self):
        """
        Returns sizes
        """
        return []

    def list_images(self, location=None, folder_ids=None):
        """
        Lists VM templates as images.
        If folder is given then it will list images contained
        in that folder only.
        """
        images = []
        if folder_ids:
            vms = []
            for folder_id in folder_ids:
                folder_object = self._get_item_by_moid('Folder', folder_id)
                vms.extend(folder_object.childEntity)
        else:
            content = self.connection.RetrieveContent()
            vms = content.viewManager.CreateContainerView(content.rootFolder, [vim.VirtualMachine], recursive=True).view
        for vm in vms:
            if vm.config and vm.config.template:
                images.append(self._to_image(vm))
        return images

    def _to_image(self, data):
        summary = data.summary
        name = summary.config.name
        uuid = summary.config.instanceUuid
        memory = summary.config.memorySizeMB
        cpus = summary.config.numCpu
        operating_system = summary.config.guestFullName
        os_type = 'unix'
        if 'Microsoft' in str(operating_system):
            os_type = 'windows'
        annotation = summary.config.annotation
        extra = {'path': summary.config.vmPathName, 'operating_system': operating_system, 'os_type': os_type, 'memory_MB': memory, 'cpus': cpus, 'overallStatus': str(summary.overallStatus), 'metadata': {}, 'type': 'template_6_5', 'disk_size': int(summary.storage.committed) // 1024 ** 3, 'datastore': data.datastore[0].info.name}
        boot_time = summary.runtime.bootTime
        if boot_time:
            extra['boot_time'] = boot_time.isoformat()
        if annotation:
            extra['annotation'] = annotation
        for custom_field in data.customValue:
            key_id = custom_field.key
            key = self.find_custom_field_key(key_id)
            extra['metadata'][key] = custom_field.value
        return NodeImage(id=uuid, name=name, driver=self, extra=extra)

    def _collect_properties(self, content, view_ref, obj_type, path_set=None, include_mors=False):
        """
        Collect properties for managed objects from a view ref
        Check the vSphere API documentation for example on retrieving
        object properties:
            - http://goo.gl/erbFDz
        Args:
            content     (ServiceInstance): ServiceInstance content
            view_ref (pyVmomi.vim.view.*): Starting point of inventory
                                           navigation
            obj_type      (pyVmomi.vim.*): Type of managed object
            path_set               (list): List of properties to retrieve
            include_mors           (bool): If True include the managed objects
                                        refs in the result
        Returns:
            A list of properties for the managed objects
        """
        collector = content.propertyCollector
        obj_spec = vmodl.query.PropertyCollector.ObjectSpec()
        obj_spec.obj = view_ref
        obj_spec.skip = True
        traversal_spec = vmodl.query.PropertyCollector.TraversalSpec()
        traversal_spec.name = 'traverseEntities'
        traversal_spec.path = 'view'
        traversal_spec.skip = False
        traversal_spec.type = view_ref.__class__
        obj_spec.selectSet = [traversal_spec]
        property_spec = vmodl.query.PropertyCollector.PropertySpec()
        property_spec.type = obj_type
        if not path_set:
            property_spec.all = True
        property_spec.pathSet = path_set
        filter_spec = vmodl.query.PropertyCollector.FilterSpec()
        filter_spec.objectSet = [obj_spec]
        filter_spec.propSet = [property_spec]
        props = collector.RetrieveContents([filter_spec])
        data = []
        for obj in props:
            properties = {}
            for prop in obj.propSet:
                properties[prop.name] = prop.val
            if include_mors:
                properties['obj'] = obj.obj
            data.append(properties)
        return data

    def list_nodes(self, enhance=True, max_properties=20):
        """
        List nodes, excluding templates
        """
        vm_properties = ['config.template', 'summary.config.name', 'summary.config.vmPathName', 'summary.config.memorySizeMB', 'summary.config.numCpu', 'summary.storage.committed', 'summary.config.guestFullName', 'summary.runtime.host', 'summary.config.instanceUuid', 'summary.config.annotation', 'summary.runtime.powerState', 'summary.runtime.bootTime', 'summary.guest.ipAddress', 'summary.overallStatus', 'customValue', 'snapshot']
        content = self.connection.RetrieveContent()
        view = content.viewManager.CreateContainerView(content.rootFolder, [vim.VirtualMachine], True)
        i = 0
        vm_dict = {}
        while i < len(vm_properties):
            vm_list = self._collect_properties(content, view, vim.VirtualMachine, path_set=vm_properties[i:i + max_properties], include_mors=True)
            i += max_properties
            for vm in vm_list:
                if not vm_dict.get(vm['obj']):
                    vm_dict[vm['obj']] = vm
                else:
                    vm_dict[vm['obj']].update(vm)
        vm_list = [vm_dict[k] for k in vm_dict]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nodes = loop.run_until_complete(self._to_nodes(vm_list))
        if enhance:
            nodes = self._enhance_metadata(nodes, content)
        return nodes

    def list_nodes_recursive(self, enhance=True):
        """
        Lists nodes, excluding templates
        """
        nodes = []
        content = self.connection.RetrieveContent()
        children = content.rootFolder.childEntity
        if content.customFieldsManager:
            self.custom_fields = content.customFieldsManager.field
        else:
            self.custom_fields = []
        for child in children:
            if hasattr(child, 'vmFolder'):
                datacenter = child
                vm_folder = datacenter.vmFolder
                vm_list = vm_folder.childEntity
                nodes.extend(self._to_nodes_recursive(vm_list))
        if enhance:
            nodes = self._enhance_metadata(nodes, content)
        return nodes

    def _enhance_metadata(self, nodes, content):
        nodemap = {}
        for node in nodes:
            node.extra['vSphere version'] = content.about.version
            nodemap[node.id] = node
        filter_spec = vim.event.EventFilterSpec(eventTypeId=['VmBeingDeployedEvent'])
        deploy_events = content.eventManager.QueryEvent(filter_spec)
        for event in deploy_events:
            try:
                uuid = event.vm.vm.config.instanceUuid
            except Exception:
                continue
            if uuid in nodemap:
                node = nodemap[uuid]
                try:
                    source_template_vm = event.srcTemplate.vm
                    image_id = source_template_vm.config.instanceUuid
                    node.extra['image_id'] = image_id
                except Exception:
                    logger.error('Cannot get instanceUuid from source template')
                try:
                    node.created_at = event.createdTime
                except AttributeError:
                    logger.error('Cannot get creation date from VM deploy event')
        return nodes

    async def _to_nodes(self, vm_list):
        vms = []
        for vm in vm_list:
            if vm.get('config.template'):
                continue
            vms.append(vm)
        loop = asyncio.get_event_loop()
        vms = [loop.run_in_executor(None, self._to_node, vms[i]) for i in range(len(vms))]
        return await asyncio.gather(*vms)

    def _to_nodes_recursive(self, vm_list):
        nodes = []
        for virtual_machine in vm_list:
            if hasattr(virtual_machine, 'childEntity'):
                nodes.extend(self._to_nodes_recursive(virtual_machine.childEntity))
            elif isinstance(virtual_machine, vim.VirtualApp):
                nodes.extend(self._to_nodes_recursive(virtual_machine.vm))
            else:
                if not hasattr(virtual_machine, 'config') or (virtual_machine.config and virtual_machine.config.template):
                    continue
                nodes.append(self._to_node_recursive(virtual_machine))
        return nodes

    def _to_node(self, vm):
        name = vm.get('summary.config.name')
        path = vm.get('summary.config.vmPathName')
        memory = vm.get('summary.config.memorySizeMB')
        cpus = vm.get('summary.config.numCpu')
        disk = vm.get('summary.storage.committed', 0) // 1024 ** 3
        id_to_hash = str(memory) + str(cpus) + str(disk)
        size_id = hashlib.md5(id_to_hash.encode('utf-8')).hexdigest()
        size_name = name + '-size'
        size_extra = {'cpus': cpus}
        driver = self
        size = NodeSize(id=size_id, name=size_name, ram=memory, disk=disk, bandwidth=0, price=0, driver=driver, extra=size_extra)
        operating_system = vm.get('summary.config.guestFullName')
        host = vm.get('summary.runtime.host')
        os_type = 'unix'
        if 'Microsoft' in str(operating_system):
            os_type = 'windows'
        uuid = vm.get('summary.config.instanceUuid') or (vm.get('obj').config and vm.get('obj').config.instanceUuid)
        if not uuid:
            logger.error('No uuid for vm: {}'.format(vm))
        annotation = vm.get('summary.config.annotation')
        state = vm.get('summary.runtime.powerState')
        status = self.NODE_STATE_MAP.get(state, NodeState.UNKNOWN)
        boot_time = vm.get('summary.runtime.bootTime')
        ip_addresses = []
        if vm.get('summary.guest.ipAddress'):
            ip_addresses.append(vm.get('summary.guest.ipAddress'))
        overall_status = str(vm.get('summary.overallStatus'))
        public_ips = []
        private_ips = []
        extra = {'path': path, 'operating_system': operating_system, 'os_type': os_type, 'memory_MB': memory, 'cpus': cpus, 'overall_status': overall_status, 'metadata': {}, 'snapshots': []}
        if disk:
            extra['disk'] = disk
        if host:
            extra['host'] = host.name
            parent = host.parent
            while parent:
                if isinstance(parent, vim.ClusterComputeResource):
                    extra['cluster'] = parent.name
                    break
                parent = parent.parent
        if boot_time:
            extra['boot_time'] = boot_time.isoformat()
        if annotation:
            extra['annotation'] = annotation
        for ip_address in ip_addresses:
            try:
                if is_public_subnet(ip_address):
                    public_ips.append(ip_address)
                else:
                    private_ips.append(ip_address)
            except Exception:
                pass
        if vm.get('snapshot'):
            extra['snapshots'] = format_snapshots(recurse_snapshots(vm.get('snapshot').rootSnapshotList))
        for custom_field in vm.get('customValue', []):
            key_id = custom_field.key
            key = self.find_custom_field_key(key_id)
            extra['metadata'][key] = custom_field.value
        node = Node(id=uuid, name=name, state=status, size=size, public_ips=public_ips, private_ips=private_ips, driver=self, extra=extra)
        node._uuid = uuid
        return node

    def _to_node_recursive(self, virtual_machine):
        summary = virtual_machine.summary
        name = summary.config.name
        path = summary.config.vmPathName
        memory = summary.config.memorySizeMB
        cpus = summary.config.numCpu
        disk = ''
        if summary.storage.committed:
            disk = summary.storage.committed / 1024 ** 3
        id_to_hash = str(memory) + str(cpus) + str(disk)
        size_id = hashlib.md5(id_to_hash.encode('utf-8')).hexdigest()
        size_name = name + '-size'
        size_extra = {'cpus': cpus}
        driver = self
        size = NodeSize(id=size_id, name=size_name, ram=memory, disk=disk, bandwidth=0, price=0, driver=driver, extra=size_extra)
        operating_system = summary.config.guestFullName
        host = summary.runtime.host
        os_type = 'unix'
        if 'Microsoft' in str(operating_system):
            os_type = 'windows'
        uuid = summary.config.instanceUuid
        annotation = summary.config.annotation
        state = summary.runtime.powerState
        status = self.NODE_STATE_MAP.get(state, NodeState.UNKNOWN)
        boot_time = summary.runtime.bootTime
        ip_addresses = []
        if summary.guest is not None:
            ip_addresses.append(summary.guest.ipAddress)
        overall_status = str(summary.overallStatus)
        public_ips = []
        private_ips = []
        extra = {'path': path, 'operating_system': operating_system, 'os_type': os_type, 'memory_MB': memory, 'cpus': cpus, 'overallStatus': overall_status, 'metadata': {}, 'snapshots': []}
        if disk:
            extra['disk'] = disk
        if host:
            extra['host'] = host.name
            parent = host.parent
            while parent:
                if isinstance(parent, vim.ClusterComputeResource):
                    extra['cluster'] = parent.name
                    break
                parent = parent.parent
        if boot_time:
            extra['boot_time'] = boot_time.isoformat()
        if annotation:
            extra['annotation'] = annotation
        for ip_address in ip_addresses:
            try:
                if is_public_subnet(ip_address):
                    public_ips.append(ip_address)
                else:
                    private_ips.append(ip_address)
            except Exception:
                pass
        if virtual_machine.snapshot:
            snapshots = [{'id': s.id, 'name': s.name, 'description': s.description, 'created': s.createTime.strftime('%Y-%m-%d %H:%M'), 'state': s.state} for s in virtual_machine.snapshot.rootSnapshotList]
            extra['snapshots'] = snapshots
        for custom_field in virtual_machine.customValue:
            key_id = custom_field.key
            key = self.find_custom_field_key(key_id)
            extra['metadata'][key] = custom_field.value
        node = Node(id=uuid, name=name, state=status, size=size, public_ips=public_ips, private_ips=private_ips, driver=self, extra=extra)
        node._uuid = uuid
        return node

    def reboot_node(self, node):
        """ """
        vm = self.find_by_uuid(node.id)
        return self.wait_for_task(vm.RebootGuest())

    def destroy_node(self, node):
        """ """
        vm = self.find_by_uuid(node.id)
        if node.state != NodeState.STOPPED:
            self.stop_node(node)
        return self.wait_for_task(vm.Destroy())

    def stop_node(self, node):
        """ """
        vm = self.find_by_uuid(node.id)
        return self.wait_for_task(vm.PowerOff())

    def start_node(self, node):
        """ """
        vm = self.find_by_uuid(node.id)
        return self.wait_for_task(vm.PowerOn())

    def ex_list_snapshots(self, node):
        """
        List node snapshots
        """
        vm = self.find_by_uuid(node.id)
        if not vm.snapshot:
            return []
        return format_snapshots(recurse_snapshots(vm.snapshot.rootSnapshotList))

    def ex_create_snapshot(self, node, snapshot_name, description='', dump_memory=False, quiesce=False):
        """
        Create node snapshot
        """
        vm = self.find_by_uuid(node.id)
        return WaitForTask(vm.CreateSnapshot(snapshot_name, description, dump_memory, quiesce))

    def ex_remove_snapshot(self, node, snapshot_name=None, remove_children=True):
        """
        Remove a snapshot from node.
        If snapshot_name is not defined remove the last one.
        """
        vm = self.find_by_uuid(node.id)
        if not vm.snapshot:
            raise LibcloudError('Remove snapshot failed. No snapshots for node %s' % node.name, driver=self)
        snapshots = recurse_snapshots(vm.snapshot.rootSnapshotList)
        if not snapshot_name:
            snapshot = snapshots[-1].snapshot
        else:
            for s in snapshots:
                if snapshot_name == s.name:
                    snapshot = s.snapshot
                    break
            else:
                raise LibcloudError('Snapshot `%s` not found' % snapshot_name, driver=self)
        return self.wait_for_task(snapshot.RemoveSnapshot_Task(removeChildren=remove_children))

    def ex_revert_to_snapshot(self, node, snapshot_name=None):
        """
        Revert node to a specific snapshot.
        If snapshot_name is not defined revert to the last one.
        """
        vm = self.find_by_uuid(node.id)
        if not vm.snapshot:
            raise LibcloudError('Revert failed. No snapshots for node %s' % node.name, driver=self)
        snapshots = recurse_snapshots(vm.snapshot.rootSnapshotList)
        if not snapshot_name:
            snapshot = snapshots[-1].snapshot
        else:
            for s in snapshots:
                if snapshot_name == s.name:
                    snapshot = s.snapshot
                    break
            else:
                raise LibcloudError('Snapshot `%s` not found' % snapshot_name, driver=self)
        return self.wait_for_task(snapshot.RevertToSnapshot_Task())

    def _find_template_by_uuid(self, template_uuid):
        try:
            template = self.find_by_uuid(template_uuid)
        except LibcloudError:
            content = self.connection.RetrieveContent()
            vms = content.viewManager.CreateContainerView(content.rootFolder, [vim.VirtualMachine], recursive=True).view
            for vm in vms:
                if vm.config.instanceUuid == template_uuid:
                    template = vm
        except Exception as exc:
            raise LibcloudError('Error while searching for template: %s' % exc, driver=self)
        if not template:
            raise LibcloudError('Unable to locate VirtualMachine.', driver=self)
        return template

    def find_by_uuid(self, node_uuid):
        """Searches VMs for a given uuid
        returns pyVmomi.VmomiSupport.vim.VirtualMachine
        """
        vm = self.connection.content.searchIndex.FindByUuid(None, node_uuid, True, True)
        if not vm:
            vm = self._get_item_by_moid('VirtualMachine', node_uuid)
            if not vm:
                raise LibcloudError('Unable to locate VirtualMachine.', driver=self)
        return vm

    def find_custom_field_key(self, key_id):
        """Return custom field key name, provided it's id"""
        if not hasattr(self, 'custom_fields'):
            content = self.connection.RetrieveContent()
            if content.customFieldsManager:
                self.custom_fields = content.customFieldsManager.field
            else:
                self.custom_fields = []
        for k in self.custom_fields:
            if k.key == key_id:
                return k.name
        return None

    def get_obj(self, vimtype, name):
        """
        Return an object by name, if name is None the
        first found object is returned
        """
        obj = None
        content = self.connection.RetrieveContent()
        container = content.viewManager.CreateContainerView(content.rootFolder, vimtype, True)
        for c in container.view:
            if name:
                if c.name == name:
                    obj = c
                    break
            else:
                obj = c
                break
        return obj

    def wait_for_task(self, task, timeout=1800, interval=10):
        """wait for a vCenter task to finish"""
        start_time = time.time()
        task_done = False
        while not task_done:
            if time.time() - start_time >= timeout:
                raise LibcloudError('Timeout while waiting for import task Id %s' % task.info.id, driver=self)
            if task.info.state == 'success':
                if task.info.result and str(task.info.result) != 'success':
                    return task.info.result
                return True
            if task.info.state == 'error':
                raise LibcloudError(task.info.error.msg, driver=self)
            time.sleep(interval)

    def create_node(self, name, image, size, location=None, ex_cluster=None, ex_network=None, ex_datacenter=None, ex_folder=None, ex_resource_pool=None, ex_datastore_cluster=None, ex_datastore=None):
        """
        Creates and returns node.

        :keyword    ex_network: Name of a "Network" to connect the VM to ",
        :type       ex_network: ``str``

        """
        template = self._find_template_by_uuid(image.id)
        if ex_cluster:
            cluster_name = ex_cluster
        else:
            cluster_name = location.name
        cluster = self.get_obj([vim.ClusterComputeResource], cluster_name)
        if not cluster:
            cluster = self.get_obj([vim.HostSystem], cluster_name)
        datacenter = None
        if not ex_datacenter:
            parent = cluster.parent
            while parent:
                if isinstance(parent, vim.Datacenter):
                    datacenter = parent
                    break
                parent = parent.parent
        if ex_datacenter or datacenter is None:
            datacenter = self.get_obj([vim.Datacenter], ex_datacenter)
        if ex_folder:
            folder = self.get_obj([vim.Folder], ex_folder)
            if folder is None:
                folder = self._get_item_by_moid('Folder', ex_folder)
        else:
            folder = datacenter.vmFolder
        if ex_resource_pool:
            resource_pool = self.get_obj([vim.ResourcePool], ex_resource_pool)
        else:
            try:
                resource_pool = cluster.resourcePool
            except AttributeError:
                resource_pool = cluster.parent.resourcePool
        devices = []
        vmconf = vim.vm.ConfigSpec(numCPUs=int(size.extra.get('cpu', 1)), memoryMB=int(size.ram), deviceChange=devices)
        datastore = None
        pod = None
        podsel = vim.storageDrs.PodSelectionSpec()
        if ex_datastore_cluster:
            pod = self.get_obj([vim.StoragePod], ex_datastore_cluster)
        else:
            content = self.connection.RetrieveContent()
            pods = content.viewManager.CreateContainerView(content.rootFolder, [vim.StoragePod], True).view
            for pod in pods:
                if cluster.name.lower() in pod.name:
                    break
        podsel.storagePod = pod
        storagespec = vim.storageDrs.StoragePlacementSpec()
        storagespec.podSelectionSpec = podsel
        storagespec.type = 'create'
        storagespec.folder = folder
        storagespec.resourcePool = resource_pool
        storagespec.configSpec = vmconf
        try:
            content = self.connection.RetrieveContent()
            rec = content.storageResourceManager.RecommendDatastores(storageSpec=storagespec)
            rec_action = rec.recommendations[0].action[0]
            real_datastore_name = rec_action.destination.name
        except Exception:
            real_datastore_name = template.datastore[0].info.name
        datastore = self.get_obj([vim.Datastore], real_datastore_name)
        if ex_datastore:
            datastore = self.get_obj([vim.Datastore], ex_datastore)
            if datastore is None:
                datastore = self._get_item_by_moid('Datastore', ex_datastore)
        elif not datastore:
            datastore = self.get_obj([vim.Datastore], template.datastore[0].info.name)
        add_network = True
        if ex_network and len(template.network) > 0:
            for nets in template.network:
                if template in nets.vm:
                    add_network = False
        if ex_network and add_network:
            nicspec = vim.vm.device.VirtualDeviceSpec()
            nicspec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
            nicspec.device = vim.vm.device.VirtualVmxnet3()
            nicspec.device.wakeOnLanEnabled = True
            nicspec.device.deviceInfo = vim.Description()
            portgroup = self.get_obj([vim.dvs.DistributedVirtualPortgroup], ex_network)
            if portgroup:
                dvs_port_connection = vim.dvs.PortConnection()
                dvs_port_connection.portgroupKey = portgroup.key
                dvs_port_connection.switchUuid = portgroup.config.distributedVirtualSwitch.uuid
                nicspec.device.backing = vim.vm.device.VirtualEthernetCard.DistributedVirtualPortBackingInfo()
                nicspec.device.backing.port = dvs_port_connection
            else:
                nicspec.device.backing = vim.vm.device.VirtualEthernetCard.NetworkBackingInfo()
                nicspec.device.backing.network = self.get_obj([vim.Network], ex_network)
                nicspec.device.backing.deviceName = ex_network
            nicspec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
            nicspec.device.connectable.startConnected = True
            nicspec.device.connectable.connected = True
            nicspec.device.connectable.allowGuestControl = True
            devices.append(nicspec)
        clonespec = vim.vm.CloneSpec(config=vmconf)
        relospec = vim.vm.RelocateSpec()
        relospec.datastore = datastore
        relospec.pool = resource_pool
        if location:
            host = self.get_obj([vim.HostSystem], location.name)
            if host:
                relospec.host = host
        clonespec.location = relospec
        clonespec.powerOn = True
        task = template.Clone(folder=folder, name=name, spec=clonespec)
        return self._to_node_recursive(self.wait_for_task(task))

    def ex_connect_network(self, vm, network_name):
        spec = vim.vm.ConfigSpec()
        dev_changes = []
        network_spec = vim.vm.device.VirtualDeviceSpec()
        network_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.add
        network_spec.device = vim.vm.device.VirtualVmxnet3()
        network_spec.device.backing = vim.vm.device.VirtualEthernetCard.NetworkBackingInfo()
        network_spec.device.backing.useAutoDetect = False
        network_spec.device.backing.network = self.get_obj([vim.Network], network_name)
        network_spec.device.connectable = vim.vm.device.VirtualDevice.ConnectInfo()
        network_spec.device.connectable.startConnected = True
        network_spec.device.connectable.connected = True
        network_spec.device.connectable.allowGuestControl = True
        dev_changes.append(network_spec)
        spec.deviceChange = dev_changes
        output = vm.ReconfigVM_Task(spec=spec)
        print(output.info)

    def _get_item_by_moid(self, type_, moid):
        vm_obj = VmomiSupport.templateOf(type_)(moid, self.connection._stub)
        return vm_obj

    def ex_list_folders(self):
        content = self.connection.RetrieveContent()
        folders_raw = content.viewManager.CreateContainerView(content.rootFolder, [vim.Folder], True).view
        folders = []
        for folder in folders_raw:
            to_add = {'type': list(folder.childType)}
            to_add['name'] = folder.name
            to_add['id'] = folder._moId
            folders.append(to_add)
        return folders

    def ex_list_datastores(self):
        content = self.connection.RetrieveContent()
        datastores_raw = content.viewManager.CreateContainerView(content.rootFolder, [vim.Datastore], True).view
        datastores = []
        for dstore in datastores_raw:
            to_add = {'type': dstore.summary.type}
            to_add['name'] = dstore.name
            to_add['id'] = dstore._moId
            to_add['free_space'] = int(dstore.summary.freeSpace)
            to_add['capacity'] = int(dstore.summary.capacity)
            datastores.append(to_add)
        return datastores

    def ex_open_console(self, vm_uuid):
        vm = self.find_by_uuid(vm_uuid)
        ticket = vm.AcquireTicket(ticketType='webmks')
        return 'wss://{}:{}/ticket/{}'.format(ticket.host, ticket.port, ticket.ticket)

    def _get_version(self):
        content = self.connection.RetrieveContent()
        return content.about.version