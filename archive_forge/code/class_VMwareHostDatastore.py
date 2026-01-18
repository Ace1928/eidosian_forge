from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, find_datastore_by_name, find_obj, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible.module_utils._text import to_native
class VMwareHostDatastore(SMS):

    def __init__(self, module):
        super(VMwareHostDatastore, self).__init__(module)
        self.datastore_name = module.params['datastore_name']
        self.datastore_type = module.params['datastore_type']
        self.nfs_server = module.params['nfs_server']
        self.nfs_path = module.params['nfs_path']
        self.nfs_ro = module.params['nfs_ro']
        self.vmfs_device_name = module.params['vmfs_device_name']
        self.vasa_provider_name = module.params['vasa_provider']
        self.vmfs_version = module.params['vmfs_version']
        self.resignature = module.params['resignature']
        self.esxi_hostname = module.params['esxi_hostname']
        self.auto_expand = module.params['auto_expand']
        self.state = module.params['state']
        if self.is_vcenter():
            if not self.esxi_hostname:
                self.module.fail_json(msg='esxi_hostname is mandatory with a vcenter')
            self.esxi = self.find_hostsystem_by_name(self.esxi_hostname)
            if self.esxi is None:
                self.module.fail_json(msg='Failed to find ESXi hostname %s' % self.esxi_hostname)
        else:
            self.esxi = find_obj(self.content, [vim.HostSystem], None)

    def process_state(self):
        ds_states = {'absent': {'present': self.umount_datastore_host, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_exit_unchanged, 'absent': self.mount_datastore_host}}
        try:
            ds_states[self.state][self.check_datastore_host_state()]()
        except (vmodl.RuntimeFault, vmodl.MethodFault) as vmodl_fault:
            self.module.fail_json(msg=to_native(vmodl_fault.msg))
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def expand_datastore_up_to_full(self):
        """
        Expand a datastore capacity up to full if there is free capacity.
        """
        cnf_mng = self.esxi.configManager
        for datastore_obj in self.esxi.datastore:
            if datastore_obj.name == self.datastore_name:
                expand_datastore_obj = datastore_obj
                break
        vmfs_ds_options = cnf_mng.datastoreSystem.QueryVmfsDatastoreExpandOptions(expand_datastore_obj)
        if vmfs_ds_options:
            if self.module.check_mode is False:
                try:
                    cnf_mng.datastoreSystem.ExpandVmfsDatastore(datastore=expand_datastore_obj, spec=vmfs_ds_options[0].spec)
                except Exception as e:
                    self.module.fail_json(msg='%s can not expand the datastore: %s' % (to_native(e.msg), self.datastore_name))
            self.module.exit_json(changed=True)

    def state_exit_unchanged(self):
        self.module.exit_json(changed=False)

    def check_datastore_host_state(self):
        storage_system = self.esxi.configManager.storageSystem
        host_file_sys_vol_mount_info = storage_system.fileSystemVolumeInfo.mountInfo
        for host_mount_info in host_file_sys_vol_mount_info:
            if host_mount_info.volume.name == self.datastore_name:
                if self.auto_expand and host_mount_info.volume.type == 'VMFS':
                    self.expand_datastore_up_to_full()
                return 'present'
        return 'absent'

    def get_used_disks_names(self):
        used_disks = []
        storage_system = self.esxi.configManager.storageSystem
        for each_vol_mount_info in storage_system.fileSystemVolumeInfo.mountInfo:
            if hasattr(each_vol_mount_info.volume, 'extent'):
                for each_partition in each_vol_mount_info.volume.extent:
                    used_disks.append(each_partition.diskName)
        return used_disks

    def umount_datastore_host(self):
        ds = find_datastore_by_name(self.content, self.datastore_name)
        if not ds:
            self.module.fail_json(msg='No datastore found with name %s' % self.datastore_name)
        if self.module.check_mode is False:
            error_message_umount = 'Cannot umount datastore %s from host %s' % (self.datastore_name, self.esxi.name)
            try:
                self.esxi.configManager.datastoreSystem.RemoveDatastore(ds)
            except (vim.fault.NotFound, vim.fault.HostConfigFault, vim.fault.ResourceInUse) as fault:
                self.module.fail_json(msg='%s: %s' % (error_message_umount, to_native(fault.msg)))
            except Exception as e:
                self.module.fail_json(msg='%s: %s' % (error_message_umount, to_native(e)))
        self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))

    def mount_datastore_host(self):
        if self.datastore_type == 'nfs' or self.datastore_type == 'nfs41':
            self.mount_nfs_datastore_host()
        if self.datastore_type == 'vmfs':
            self.mount_vmfs_datastore_host()
        if self.datastore_type == 'vvol':
            self.mount_vvol_datastore_host()

    def mount_nfs_datastore_host(self):
        if self.module.check_mode is False:
            mnt_specs = vim.host.NasVolume.Specification()
            if self.datastore_type == 'nfs':
                mnt_specs.type = 'NFS'
                mnt_specs.remoteHost = self.nfs_server
            if self.datastore_type == 'nfs41':
                mnt_specs.type = 'NFS41'
                mnt_specs.remoteHost = 'something'
                mnt_specs.remoteHostNames = [self.nfs_server]
            mnt_specs.remotePath = self.nfs_path
            mnt_specs.localPath = self.datastore_name
            if self.nfs_ro:
                mnt_specs.accessMode = 'readOnly'
            else:
                mnt_specs.accessMode = 'readWrite'
            error_message_mount = 'Cannot mount datastore %s on host %s' % (self.datastore_name, self.esxi.name)
            try:
                ds = self.esxi.configManager.datastoreSystem.CreateNasDatastore(mnt_specs)
                if not ds:
                    self.module.fail_json(msg=error_message_mount)
            except (vim.fault.NotFound, vim.fault.DuplicateName, vim.fault.AlreadyExists, vim.fault.HostConfigFault, vmodl.fault.InvalidArgument, vim.fault.NoVirtualNic, vim.fault.NoGateway) as fault:
                self.module.fail_json(msg='%s: %s' % (error_message_mount, to_native(fault.msg)))
            except Exception as e:
                self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(e)))
        self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))

    def mount_vmfs_datastore_host(self):
        if self.module.check_mode is False:
            ds_path = '/vmfs/devices/disks/' + str(self.vmfs_device_name)
            host_ds_system = self.esxi.configManager.datastoreSystem
            ds_system = vim.host.DatastoreSystem
            if self.vmfs_device_name in self.get_used_disks_names():
                error_message_used_disk = 'VMFS disk %s already in use' % self.vmfs_device_name
                self.module.fail_json(msg='%s' % error_message_used_disk)
            error_message_mount = 'Cannot mount datastore %s on host %s' % (self.datastore_name, self.esxi.name)
            try:
                if self.resignature:
                    storage_system = self.esxi.configManager.storageSystem
                    host_unres_volumes = storage_system.QueryUnresolvedVmfsVolume()
                    unres_vol_extents = {}
                    for unres_vol in host_unres_volumes:
                        for ext in unres_vol.extent:
                            unres_vol_extents[ext.device.diskName] = ext
                    if self.vmfs_device_name in unres_vol_extents:
                        spec = vim.host.UnresolvedVmfsResignatureSpec()
                        spec.extentDevicePath = unres_vol_extents[self.vmfs_device_name].devicePath
                        task = host_ds_system.ResignatureUnresolvedVmfsVolume_Task(spec)
                        wait_for_task(task=task)
                        task.info.result.result.RenameDatastore(self.datastore_name)
                else:
                    vmfs_ds_options = ds_system.QueryVmfsDatastoreCreateOptions(host_ds_system, ds_path, self.vmfs_version)
                    vmfs_ds_options[0].spec.vmfs.volumeName = self.datastore_name
                    ds_system.CreateVmfsDatastore(host_ds_system, vmfs_ds_options[0].spec)
            except (vim.fault.NotFound, vim.fault.DuplicateName, vim.fault.HostConfigFault, vmodl.fault.InvalidArgument) as fault:
                self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(fault.msg)))
            except Exception as e:
                self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(e)))
        self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))

    def mount_vvol_datastore_host(self):
        if self.module.check_mode is False:
            self.get_sms_connection()
            storage_manager = self.sms_si.QueryStorageManager()
            container_result = storage_manager.QueryStorageContainer()
            provider = None
            for p in container_result.providerInfo:
                if p.name == self.vasa_provider_name:
                    provider = p
                    break
            if provider is None:
                error_message_provider = 'VASA provider %s not found' % self.vasa_provider_name
                self.module.fail_json(msg='%s' % error_message_provider)
            container = None
            for sc in container_result.storageContainer:
                if sc.providerId[0] == provider.uid:
                    container = sc
                    break
            if container is None:
                error_message_container = 'vVol container for provider %s not found' % provider.uid
                self.module.fail_json(msg='%s' % error_message_container)
            vvol_spec = vim.HostDatastoreSystem.VvolDatastoreSpec(name=self.datastore_name, scId=container.uuid)
            host_ds_system = self.esxi.configManager.datastoreSystem
            error_message_mount = 'Cannot mount datastore %s on host %s' % (self.datastore_name, self.esxi.name)
            try:
                host_ds_system.CreateVvolDatastore(vvol_spec)
            except Exception as e:
                self.module.fail_json(msg='%s : %s' % (error_message_mount, to_native(e)))
        self.module.exit_json(changed=True, result='Datastore %s on host %s' % (self.datastore_name, self.esxi.name))