from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
class PyVmomi(object):

    def __init__(self, module):
        """
        Constructor
        """
        if not HAS_REQUESTS:
            module.fail_json(msg=missing_required_lib('requests'), exception=REQUESTS_IMP_ERR)
        if not HAS_PYVMOMI:
            module.fail_json(msg=missing_required_lib('PyVmomi'), exception=PYVMOMI_IMP_ERR)
        self.module = module
        self.params = module.params
        self.current_vm_obj = None
        self.si, self.content = connect_to_api(self.module, return_si=True)
        self.custom_field_mgr = []
        if self.content.customFieldsManager:
            self.custom_field_mgr = self.content.customFieldsManager.field

    def is_vcenter(self):
        """
        Check if given hostname is vCenter or ESXi host
        Returns: True if given connection is with vCenter server
                 False if given connection is with ESXi server

        """
        api_type = None
        try:
            api_type = self.content.about.apiType
        except (vmodl.RuntimeFault, vim.fault.VimFault) as exc:
            self.module.fail_json(msg='Failed to get status of vCenter server : %s' % exc.msg)
        if api_type == 'VirtualCenter':
            return True
        elif api_type == 'HostAgent':
            return False

    def vcenter_version_at_least(self, version=None):
        """
        Check that the vCenter server is at least a specific version number
        Args:
            version (tuple): a version tuple, for example (6, 7, 0)
        Returns: bool
        """
        if version:
            vc_version = self.content.about.version
            return StrictVersion(vc_version) >= StrictVersion('.'.join(map(str, version)))
        self.module.fail_json(msg='The passed vCenter version: %s is None.' % version)

    def get_cert_fingerprint(self, fqdn, port, proxy_host=None, proxy_port=None):
        if proxy_host:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((proxy_host, proxy_port))
            command = 'CONNECT %s:%d HTTP/1.0\r\n\r\n' % (fqdn, port)
            sock.send(command.encode())
            buf = sock.recv(8192).decode()
            if buf.split()[1] != '200':
                self.module.fail_json(msg='Failed to connect to the proxy')
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            der_cert_bin = ctx.wrap_socket(sock, server_hostname=fqdn).getpeercert(True)
            sock.close()
        else:
            try:
                pem = ssl.get_server_certificate((fqdn, port))
            except Exception:
                self.module.fail_json(msg=f'Cannot connect to host: {fqdn}')
            der_cert_bin = ssl.PEM_cert_to_DER_cert(pem)
        if der_cert_bin:
            string = str(hashlib.sha1(der_cert_bin).hexdigest())
            return ':'.join((a + b for a, b in zip(string[::2], string[1::2])))
        else:
            self.module.fail_json(msg=f'Unable to obtain certificate fingerprint for host: {fqdn}')

    def get_managed_objects_properties(self, vim_type, properties=None):
        """
        Look up a Managed Object Reference in vCenter / ESXi Environment
        :param vim_type: Type of vim object e.g, for datacenter - vim.Datacenter
        :param properties: List of properties related to vim object e.g. Name
        :return: local content object
        """
        root_folder = self.content.rootFolder
        if properties is None:
            properties = ['name']
        mor = self.content.viewManager.CreateContainerView(root_folder, [vim_type], True)
        traversal_spec = vmodl.query.PropertyCollector.TraversalSpec(name='traversal_spec', path='view', skip=False, type=vim.view.ContainerView)
        property_spec = vmodl.query.PropertyCollector.PropertySpec(type=vim_type, all=False, pathSet=properties)
        object_spec = vmodl.query.PropertyCollector.ObjectSpec(obj=mor, skip=True, selectSet=[traversal_spec])
        filter_spec = vmodl.query.PropertyCollector.FilterSpec(objectSet=[object_spec], propSet=[property_spec], reportMissingObjectsInResults=False)
        return self.content.propertyCollector.RetrieveContents([filter_spec])

    def get_vm(self):
        """
        Find unique virtual machine either by UUID, MoID or Name.
        Returns: virtual machine object if found, else None.

        """
        vm_obj = None
        user_desired_path = None
        use_instance_uuid = self.params.get('use_instance_uuid') or False
        if 'uuid' in self.params and self.params['uuid']:
            if not use_instance_uuid:
                vm_obj = find_vm_by_id(self.content, vm_id=self.params['uuid'], vm_id_type='uuid')
            elif use_instance_uuid:
                vm_obj = find_vm_by_id(self.content, vm_id=self.params['uuid'], vm_id_type='instance_uuid')
        elif 'name' in self.params and self.params['name']:
            objects = self.get_managed_objects_properties(vim_type=vim.VirtualMachine, properties=['name'])
            vms = []
            for temp_vm_object in objects:
                if len(temp_vm_object.propSet) == 1 and unquote(temp_vm_object.propSet[0].val) == self.params['name']:
                    vms.append(temp_vm_object.obj)
            if len(vms) > 1:
                if self.params['folder'] is None:
                    self.module.fail_json(msg='Multiple virtual machines with same name [%s] found, Folder value is a required parameter to find uniqueness of the virtual machine' % self.params['name'], details='Please see documentation of the vmware_guest module for folder parameter.')
                user_folder = self.params['folder']
                user_defined_dc = self.params['datacenter']
                datacenter_obj = find_datacenter_by_name(self.content, self.params['datacenter'])
                dcpath = compile_folder_path_for_object(vobj=datacenter_obj)
                if not dcpath.endswith('/'):
                    dcpath += '/'
                if user_folder in [None, '', '/']:
                    self.module.fail_json(msg="vmware_guest found multiple virtual machines with same name [%s], please specify folder path other than blank or '/'" % self.params['name'])
                elif user_folder.startswith('/vm/'):
                    user_desired_path = '%s%s%s' % (dcpath, user_defined_dc, user_folder)
                else:
                    user_desired_path = user_folder
                for vm in vms:
                    actual_vm_folder_path = self.get_vm_path(content=self.content, vm_name=vm)
                    if not actual_vm_folder_path.startswith('%s%s' % (dcpath, user_defined_dc)):
                        continue
                    if user_desired_path in actual_vm_folder_path:
                        vm_obj = vm
                        break
            elif vms:
                actual_vm_folder_path = self.get_vm_path(content=self.content, vm_name=vms[0])
                if self.params.get('folder') is None:
                    vm_obj = vms[0]
                elif self.params['folder'] in actual_vm_folder_path:
                    vm_obj = vms[0]
        elif 'moid' in self.params and self.params['moid']:
            vm_obj = VmomiSupport.templateOf('VirtualMachine')(self.params['moid'], self.si._stub)
            try:
                getattr(vm_obj, 'name')
            except vmodl.fault.ManagedObjectNotFound:
                vm_obj = None
        if vm_obj:
            self.current_vm_obj = vm_obj
        return vm_obj

    def gather_facts(self, vm):
        """
        Gather facts of virtual machine.
        Args:
            vm: Name of virtual machine.

        Returns: Facts dictionary of the given virtual machine.

        """
        return gather_vm_facts(self.content, vm)

    @staticmethod
    def get_vm_path(content, vm_name):
        """
        Find the path of virtual machine.
        Args:
            content: VMware content object
            vm_name: virtual machine managed object

        Returns: Folder of virtual machine if exists, else None

        """
        folder_name = None
        folder = vm_name.parent
        if folder:
            folder_name = folder.name
            fp = folder.parent
            while fp is not None and fp.name is not None and (fp != content.rootFolder):
                folder_name = fp.name + '/' + folder_name
                try:
                    fp = fp.parent
                except Exception:
                    break
            folder_name = '/' + folder_name
        return folder_name

    def get_vm_or_template(self, template_name=None):
        """
        Find the virtual machine or virtual machine template using name
        used for cloning purpose.
        Args:
            template_name: Name of virtual machine or virtual machine template

        Returns: virtual machine or virtual machine template object

        """
        template_obj = None
        if not template_name:
            return template_obj
        if '/' in template_name:
            vm_obj_path = os.path.dirname(template_name)
            vm_obj_name = os.path.basename(template_name)
            template_obj = find_vm_by_id(self.content, vm_obj_name, vm_id_type='inventory_path', folder=vm_obj_path)
            if template_obj:
                return template_obj
        else:
            template_obj = find_vm_by_id(self.content, vm_id=template_name, vm_id_type='uuid')
            if template_obj:
                return template_obj
            objects = self.get_managed_objects_properties(vim_type=vim.VirtualMachine, properties=['name'])
            templates = []
            for temp_vm_object in objects:
                if len(temp_vm_object.propSet) != 1:
                    continue
                for temp_vm_object_property in temp_vm_object.propSet:
                    if temp_vm_object_property.val == template_name:
                        templates.append(temp_vm_object.obj)
                        break
            if len(templates) > 1:
                self.module.fail_json(msg='Multiple virtual machines or templates with same name [%s] found.' % template_name)
            elif templates:
                template_obj = templates[0]
        return template_obj

    def find_cluster_by_name(self, cluster_name, datacenter_name=None):
        """
        Find Cluster by name in given datacenter
        Args:
            cluster_name: Name of cluster name to find
            datacenter_name: (optional) Name of datacenter

        Returns: True if found

        """
        return find_cluster_by_name(self.content, cluster_name, datacenter=datacenter_name)

    def get_all_hosts_by_cluster(self, cluster_name):
        """
        Get all hosts from cluster by cluster name
        Args:
            cluster_name: Name of cluster

        Returns: List of hosts

        """
        cluster_obj = self.find_cluster_by_name(cluster_name=cluster_name)
        if cluster_obj:
            return list(cluster_obj.host)
        else:
            return []

    def find_hostsystem_by_name(self, host_name, datacenter=None):
        """
        Find Host by name
        Args:
            host_name: Name of ESXi host
            datacenter: (optional) Datacenter of ESXi resides

        Returns: True if found

        """
        return find_hostsystem_by_name(self.content, hostname=host_name, datacenter=datacenter)

    def get_all_host_objs(self, cluster_name=None, esxi_host_name=None):
        """
        Get all host system managed object

        Args:
            cluster_name: Name of Cluster
            esxi_host_name: Name of ESXi server

        Returns: A list of all host system managed objects, else empty list

        """
        host_obj_list = []
        if not self.is_vcenter():
            hosts = get_all_objs(self.content, [vim.HostSystem]).keys()
            if hosts:
                host_obj_list.append(list(hosts)[0])
        elif cluster_name:
            cluster_obj = self.find_cluster_by_name(cluster_name=cluster_name)
            if cluster_obj:
                host_obj_list = list(cluster_obj.host)
            else:
                self.module.fail_json(changed=False, msg="Cluster '%s' not found" % cluster_name)
        elif esxi_host_name:
            if isinstance(esxi_host_name, str):
                esxi_host_name = [esxi_host_name]
            for host in esxi_host_name:
                esxi_host_obj = self.find_hostsystem_by_name(host_name=host)
                if esxi_host_obj:
                    host_obj_list.append(esxi_host_obj)
                else:
                    self.module.fail_json(changed=False, msg="ESXi '%s' not found" % host)
        return host_obj_list

    def host_version_at_least(self, version=None, vm_obj=None, host_name=None):
        """
        Check that the ESXi Host is at least a specific version number
        Args:
            vm_obj: virtual machine object, required one of vm_obj, host_name
            host_name (string): ESXi host name
            version (tuple): a version tuple, for example (6, 7, 0)
        Returns: bool
        """
        if vm_obj:
            host_system = vm_obj.summary.runtime.host
        elif host_name:
            host_system = self.find_hostsystem_by_name(host_name=host_name)
        else:
            self.module.fail_json(msg='VM object or ESXi host name must be set one.')
        if host_system and version:
            host_version = host_system.summary.config.product.version
            return StrictVersion(host_version) >= StrictVersion('.'.join(map(str, version)))
        else:
            self.module.fail_json(msg='Unable to get the ESXi host from vm: %s, or hostname %s,or the passed ESXi version: %s is None.' % (vm_obj, host_name, version))

    @staticmethod
    def find_host_portgroup_by_name(host, portgroup_name):
        """
        Find Portgroup on given host
        Args:
            host: Host config object
            portgroup_name: Name of portgroup

        Returns: True if found else False

        """
        for portgroup in host.config.network.portgroup:
            if portgroup.spec.name == portgroup_name:
                return portgroup
        return False

    def get_all_port_groups_by_host(self, host_system):
        """
        Get all Port Group by host
        Args:
            host_system: Name of Host System

        Returns: List of Port Group Spec
        """
        pgs_list = []
        for pg in host_system.config.network.portgroup:
            pgs_list.append(pg)
        return pgs_list

    def find_network_by_name(self, network_name=None):
        """
        Get network specified by name
        Args:
            network_name: Name of network

        Returns: List of network managed objects
        """
        networks = []
        if not network_name:
            return networks
        objects = self.get_managed_objects_properties(vim_type=vim.Network, properties=['name'])
        for temp_vm_object in objects:
            if len(temp_vm_object.propSet) != 1:
                continue
            for temp_vm_object_property in temp_vm_object.propSet:
                if temp_vm_object_property.val == network_name:
                    networks.append(temp_vm_object.obj)
                    break
        return networks

    def network_exists_by_name(self, network_name=None):
        """
        Check if network with a specified name exists or not
        Args:
            network_name: Name of network

        Returns: True if network exists else False
        """
        ret = False
        if not network_name:
            return ret
        ret = True if self.find_network_by_name(network_name=network_name) else False
        return ret

    def find_datacenter_by_name(self, datacenter_name):
        """
        Get datacenter managed object by name

        Args:
            datacenter_name: Name of datacenter

        Returns: datacenter managed object if found else None

        """
        return find_datacenter_by_name(self.content, datacenter_name=datacenter_name)

    def is_datastore_valid(self, datastore_obj=None):
        """
        Check if datastore selected is valid or not
        Args:
            datastore_obj: datastore managed object

        Returns: True if datastore is valid, False if not
        """
        if not datastore_obj or datastore_obj.summary.maintenanceMode != 'normal' or (not datastore_obj.summary.accessible):
            return False
        return True

    def find_datastore_by_name(self, datastore_name, datacenter_name=None):
        """
        Get datastore managed object by name
        Args:
            datastore_name: Name of datastore
            datacenter_name: Name of datacenter where the datastore resides.  This is needed because Datastores can be
            shared across Datacenters, so we need to specify the datacenter to assure we get the correct Managed Object Reference

        Returns: datastore managed object if found else None

        """
        return find_datastore_by_name(self.content, datastore_name=datastore_name, datacenter_name=datacenter_name)

    def find_folder_by_name(self, folder_name):
        """
        Get vm folder managed object by name
        Args:
            folder_name: Name of the vm folder

        Returns: vm folder managed object if found else None

        """
        return find_folder_by_name(self.content, folder_name=folder_name)

    def find_folder_by_fqpn(self, folder_name, datacenter_name=None, folder_type=None):
        """
        Get a unique folder managed object by specifying its Fully Qualified Path Name
        as datacenter/folder_type/sub1/sub2
        Args:
            folder_name: Fully Qualified Path Name folder name
            datacenter_name: Name of the datacenter, taken from Fully Qualified Path Name if not defined
            folder_type: Type of folder, vm, host, datastore or network,
                         taken from Fully Qualified Path Name if not defined

        Returns: folder managed object if found, else None

        """
        return find_folder_by_fqpn(self.content, folder_name=folder_name, datacenter_name=datacenter_name, folder_type=folder_type)

    def find_datastore_cluster_by_name(self, datastore_cluster_name, datacenter=None, folder=None):
        """
        Get datastore cluster managed object by name
        Args:
            datastore_cluster_name: Name of datastore cluster
            datacenter: Managed object of the datacenter
            folder: Managed object of the folder which holds datastore

        Returns: Datastore cluster managed object if found else None

        """
        if datacenter and hasattr(datacenter, 'datastoreFolder'):
            folder = datacenter.datastoreFolder
        if not folder:
            folder = self.content.rootFolder
        data_store_clusters = get_all_objs(self.content, [vim.StoragePod], folder=folder)
        for dsc in data_store_clusters:
            if dsc.name == datastore_cluster_name:
                return dsc
        return None

    def get_recommended_datastore(self, datastore_cluster_obj=None):
        """
        Return Storage DRS recommended datastore from datastore cluster
        Args:
            datastore_cluster_obj: datastore cluster managed object

        Returns: Name of recommended datastore from the given datastore cluster

        """
        if datastore_cluster_obj is None:
            return None
        sdrs_status = datastore_cluster_obj.podStorageDrsEntry.storageDrsConfig.podConfig.enabled
        if sdrs_status:
            pod_sel_spec = vim.storageDrs.PodSelectionSpec()
            pod_sel_spec.storagePod = datastore_cluster_obj
            storage_spec = vim.storageDrs.StoragePlacementSpec()
            storage_spec.podSelectionSpec = pod_sel_spec
            storage_spec.type = 'create'
            try:
                rec = self.content.storageResourceManager.RecommendDatastores(storageSpec=storage_spec)
                rec_action = rec.recommendations[0].action[0]
                return rec_action.destination.name
            except Exception:
                pass
        datastore = None
        datastore_freespace = 0
        for ds in datastore_cluster_obj.childEntity:
            if isinstance(ds, vim.Datastore) and ds.summary.freeSpace > datastore_freespace:
                if not self.is_datastore_valid(datastore_obj=ds):
                    continue
                datastore = ds
                datastore_freespace = ds.summary.freeSpace
        if datastore:
            return datastore.name
        return None

    def find_resource_pool_by_name(self, resource_pool_name='Resources', folder=None):
        """
        Get resource pool managed object by name
        Args:
            resource_pool_name: Name of resource pool

        Returns: Resource pool managed object if found else None

        """
        if not folder:
            folder = self.content.rootFolder
        resource_pools = get_all_objs(self.content, [vim.ResourcePool], folder=folder)
        for rp in resource_pools:
            if rp.name == resource_pool_name:
                return rp
        return None

    def find_resource_pool_by_cluster(self, resource_pool_name='Resources', cluster=None):
        """
        Get resource pool managed object by cluster object
        Args:
            resource_pool_name: Name of resource pool
            cluster: Managed object of cluster

        Returns: Resource pool managed object if found else None

        """
        desired_rp = None
        if not cluster:
            return desired_rp
        if resource_pool_name != 'Resources':
            resource_pools = cluster.resourcePool.resourcePool
            if resource_pools:
                for rp in resource_pools:
                    if rp.name == resource_pool_name:
                        desired_rp = rp
                        break
        else:
            desired_rp = cluster.resourcePool
        return desired_rp

    def vmdk_disk_path_split(self, vmdk_path):
        """
        Takes a string in the format

            [datastore_name] path/to/vm_name.vmdk

        Returns a tuple with multiple strings:

        1. datastore_name: The name of the datastore (without brackets)
        2. vmdk_fullpath: The "path/to/vm_name.vmdk" portion
        3. vmdk_filename: The "vm_name.vmdk" portion of the string (os.path.basename equivalent)
        4. vmdk_folder: The "path/to/" portion of the string (os.path.dirname equivalent)
        """
        try:
            datastore_name = re.match('^\\[(.*?)\\]', vmdk_path, re.DOTALL).groups()[0]
            vmdk_fullpath = re.match('\\[.*?\\] (.*)$', vmdk_path).groups()[0]
            vmdk_filename = os.path.basename(vmdk_fullpath)
            vmdk_folder = os.path.dirname(vmdk_fullpath)
            return (datastore_name, vmdk_fullpath, vmdk_filename, vmdk_folder)
        except (IndexError, AttributeError) as e:
            self.module.fail_json(msg="Bad path '%s' for filename disk vmdk image: %s" % (vmdk_path, to_native(e)))

    def find_vmdk_file(self, datastore_obj, vmdk_fullpath, vmdk_filename, vmdk_folder):
        """
        Return vSphere file object or fail_json
        Args:
            datastore_obj: Managed object of datastore
            vmdk_fullpath: Path of VMDK file e.g., path/to/vm/vmdk_filename.vmdk
            vmdk_filename: Name of vmdk e.g., VM0001_1.vmdk
            vmdk_folder: Base dir of VMDK e.g, path/to/vm

        """
        browser = datastore_obj.browser
        datastore_name = datastore_obj.name
        datastore_name_sq = '[' + datastore_name + ']'
        if browser is None:
            self.module.fail_json(msg='Unable to access browser for datastore %s' % datastore_name)
        detail_query = vim.host.DatastoreBrowser.FileInfo.Details(fileOwner=True, fileSize=True, fileType=True, modification=True)
        search_spec = vim.host.DatastoreBrowser.SearchSpec(details=detail_query, matchPattern=[vmdk_filename], searchCaseInsensitive=True)
        search_res = browser.SearchSubFolders(datastorePath=datastore_name_sq, searchSpec=search_spec)
        changed = False
        vmdk_path = datastore_name_sq + ' ' + vmdk_fullpath
        try:
            changed, result = wait_for_task(search_res)
        except TaskError as task_e:
            self.module.fail_json(msg=to_native(task_e))
        if not changed:
            self.module.fail_json(msg='No valid disk vmdk image found for path %s' % vmdk_path)
        target_folder_paths = [datastore_name_sq + ' ' + vmdk_folder + '/', datastore_name_sq + ' ' + vmdk_folder]
        for file_result in search_res.info.result:
            for f in getattr(file_result, 'file'):
                if f.path == vmdk_filename and file_result.folderPath in target_folder_paths:
                    return f
        self.module.fail_json(msg='No vmdk file found for path specified [%s]' % vmdk_path)

    def find_first_class_disk_by_name(self, disk_name, datastore_obj):
        """
        Get first-class disk managed object by name
        Args:
            disk_name: Name of the first-class disk
            datastore_obj: Managed object of datastore

        Returns: First-class disk managed object if found else None

        """
        if self.is_vcenter():
            for id in self.content.vStorageObjectManager.ListVStorageObject(datastore_obj):
                disk = self.content.vStorageObjectManager.RetrieveVStorageObject(id, datastore_obj)
                if disk.config.name == disk_name:
                    return disk
        else:
            for id in self.content.vStorageObjectManager.HostListVStorageObject(datastore_obj):
                disk = self.content.vStorageObjectManager.HostRetrieveVStorageObject(id, datastore_obj)
                if disk.config.name == disk_name:
                    return disk
        return None

    def find_first_class_disks(self, datastore_obj):
        """
        Get first-class disks managed object
        Args:
            datastore_obj: Managed object of datastore

        Returns: First-class disks managed object if found else None

        """
        disks = []
        if self.is_vcenter():
            for id in self.content.vStorageObjectManager.ListVStorageObject(datastore_obj):
                disks.append(self.content.vStorageObjectManager.RetrieveVStorageObject(id, datastore_obj))
        else:
            for id in self.content.vStorageObjectManager.HostListVStorageObject(datastore_obj):
                disks.append(self.content.vStorageObjectManager.HostRetrieveVStorageObject(id, datastore_obj))
        if disks == []:
            return None
        else:
            return disks

    def _deepmerge(self, d, u):
        """
        Deep merges u into d.

        Credit:
          https://bit.ly/2EDOs1B (stackoverflow question 3232943)
        License:
          cc-by-sa 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)
        Changes:
          using collections_compat for compatibility

        Args:
          - d (dict): dict to merge into
          - u (dict): dict to merge into d

        Returns:
          dict, with u merged into d
        """
        for k, v in iteritems(u):
            if isinstance(v, collections_compat.Mapping):
                d[k] = self._deepmerge(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _extract(self, data, remainder):
        """
        This is used to break down dotted properties for extraction.

        Args:
          - data (dict): result of _jsonify on a property
          - remainder: the remainder of the dotted property to select

        Return:
          dict
        """
        result = dict()
        if '.' not in remainder:
            result[remainder] = data[remainder]
            return result
        key, remainder = remainder.split('.', 1)
        if isinstance(data, list):
            temp_ds = []
            for i in range(len(data)):
                temp_ds.append(self._extract(data[i][key], remainder))
            result[key] = temp_ds
        else:
            result[key] = self._extract(data[key], remainder)
        return result

    def _jsonify(self, obj):
        """
        Convert an object from pyVmomi into JSON.

        Args:
          - obj (object): vim object

        Return:
          dict
        """
        return json.loads(json.dumps(obj, cls=VmomiSupport.VmomiJSONEncoder, sort_keys=True, strip_dynamic=True))

    def to_json(self, obj, properties=None):
        """
        Convert a vSphere (pyVmomi) Object into JSON.  This is a deep
        transformation.  The list of properties is optional - if not
        provided then all properties are deeply converted.  The resulting
        JSON is sorted to improve human readability.

        Requires upstream support from pyVmomi > 6.7.1
        (https://github.com/vmware/pyvmomi/pull/732)

        Args:
          - obj (object): vim object
          - properties (list, optional): list of properties following
                the property collector specification, for example:
                ["config.hardware.memoryMB", "name", "overallStatus"]
                default is a complete object dump, which can be large

        Return:
          dict
        """
        if not HAS_PYVMOMIJSON:
            self.module.fail_json(msg='The installed version of pyvmomi lacks JSON output support; need pyvmomi>6.7.1')
        result = dict()
        if properties:
            for prop in properties:
                try:
                    if '.' in prop:
                        key, remainder = prop.split('.', 1)
                        tmp = dict()
                        tmp[key] = self._extract(self._jsonify(getattr(obj, key)), remainder)
                        self._deepmerge(result, tmp)
                    else:
                        result[prop] = self._jsonify(getattr(obj, prop))
                        prop_name = prop
                        if prop.lower() == '_moid':
                            prop_name = 'moid'
                        elif prop.lower() == '_vimref':
                            prop_name = 'vimref'
                        result[prop_name] = result[prop]
                except (AttributeError, KeyError):
                    self.module.fail_json(msg="Property '{0}' not found.".format(prop))
        else:
            result = self._jsonify(obj)
        return result

    def get_folder_path(self, cur):
        full_path = '/' + cur.name
        while hasattr(cur, 'parent') and cur.parent:
            if cur.parent == self.content.rootFolder:
                break
            cur = cur.parent
            full_path = '/' + cur.name + full_path
        return full_path

    def find_obj_by_moid(self, object_type, moid):
        """
        Get Managed Object based on an object type and moid.
        If you'd like to search for a virtual machine, recommended you use get_vm method.

        Args:
          - object_type: Managed Object type
                It is possible to specify types the following.
                ["Datacenter", "ClusterComputeResource", "ResourcePool", "Folder", "HostSystem",
                 "VirtualMachine", "DistributedVirtualSwitch", "DistributedVirtualPortgroup", "Datastore"]
          - moid: moid of Managed Object
        :return: Managed Object if it exists else None
        """
        obj = VmomiSupport.templateOf(object_type)(moid, self.si._stub)
        try:
            getattr(obj, 'name')
        except vmodl.fault.ManagedObjectNotFound:
            obj = None
        return obj