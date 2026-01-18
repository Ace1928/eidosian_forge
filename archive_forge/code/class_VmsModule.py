from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class VmsModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super(VmsModule, self).__init__(*args, **kwargs)
        self._initialization = None
        self._is_new = False

    def __get_template_with_version(self):
        """
        oVirt/RHV in version 4.1 doesn't support search by template+version_number,
        so we need to list all templates with specific name and then iterate
        through it's version until we find the version we look for.
        """
        template = None
        templates_service = self._connection.system_service().templates_service()
        if self._is_new:
            if self.param('template'):
                clusters_service = self._connection.system_service().clusters_service()
                cluster = search_by_name(clusters_service, self.param('cluster'))
                data_center = self._connection.follow_link(cluster.data_center)
                templates = templates_service.list(search='name=%s and datacenter=%s' % (self.param('template'), data_center.name))
                if self.param('template_version'):
                    templates = [t for t in templates if t.version.version_number == self.param('template_version')]
                if not templates:
                    raise ValueError("Template with name '%s' and version '%s' in data center '%s' was not found" % (self.param('template'), self.param('template_version'), data_center.name))
                template = sorted(templates, key=lambda t: t.version.version_number, reverse=True)[0]
            else:
                template = templates_service.template_service('00000000-0000-0000-0000-000000000000').get()
        else:
            templates = templates_service.list(search='vm.name=%s' % self.param('name'))
            if templates:
                template = templates[0]
                if self.param('template') is not None and self.param('template') != template.name:
                    raise ValueError('You can not change template of the Virtual Machine.')
        return template

    def __get_storage_domain_and_all_template_disks(self, template):
        if self.param('template') is None:
            return None
        if self.param('storage_domain') is None:
            return None
        disks = list()
        for att in self._connection.follow_link(template.disk_attachments):
            disks.append(otypes.DiskAttachment(disk=otypes.Disk(id=att.disk.id, format=otypes.DiskFormat(self.param('disk_format')), sparse=self.param('disk_format') != 'raw', storage_domains=[otypes.StorageDomain(id=get_id_by_name(self._connection.system_service().storage_domains_service(), self.param('storage_domain')))])))
        return disks

    def __get_snapshot(self):
        if self.param('snapshot_vm') is None:
            return None
        if self.param('snapshot_name') is None:
            return None
        vms_service = self._connection.system_service().vms_service()
        vm_id = get_id_by_name(vms_service, self.param('snapshot_vm'))
        vm_service = vms_service.vm_service(vm_id)
        snaps_service = vm_service.snapshots_service()
        snaps = snaps_service.list()
        snap = next((s for s in snaps if s.description == self.param('snapshot_name')), None)
        if not snap:
            raise ValueError('Snapshot with the name "{0}" was not found.'.format(self.param('snapshot_name')))
        return snap

    def __get_placement_policy(self):
        hosts = None
        if self.param('placement_policy_hosts'):
            hosts = [otypes.Host(name=host) for host in self.param('placement_policy_hosts')]
        elif self.param('host'):
            hosts = [otypes.Host(name=self.param('host'))]
        if self.param('placement_policy'):
            return otypes.VmPlacementPolicy(affinity=otypes.VmAffinity(self.param('placement_policy')), hosts=hosts)
        return None

    def __get_cluster(self):
        if self.param('cluster') is not None:
            return self.param('cluster')
        elif self.param('snapshot_name') is not None and self.param('snapshot_vm') is not None:
            vms_service = self._connection.system_service().vms_service()
            vm = search_by_name(vms_service, self.param('snapshot_vm'))
            return self._connection.system_service().clusters_service().cluster_service(vm.cluster.id).get().name

    def build_entity(self):
        template = self.__get_template_with_version()
        cluster = self.__get_cluster()
        snapshot = self.__get_snapshot()
        placement_policy = self.__get_placement_policy()
        display = self.param('graphical_console') or dict()
        disk_attachments = self.__get_storage_domain_and_all_template_disks(template)
        return otypes.Vm(id=self.param('id'), name=self.param('name'), cluster=otypes.Cluster(name=cluster) if cluster else None, disk_attachments=disk_attachments, template=otypes.Template(id=template.id) if template else None, use_latest_template_version=self.param('use_latest_template_version'), stateless=self.param('stateless') or self.param('use_latest_template_version'), delete_protected=self.param('delete_protected'), custom_emulated_machine=self.param('custom_emulated_machine'), bios=otypes.Bios(boot_menu=otypes.BootMenu(enabled=self.param('boot_menu')) if self.param('boot_menu') is not None else None, type=otypes.BiosType[self.param('bios_type').upper()] if self.param('bios_type') is not None else None) if self.param('boot_menu') is not None or self.param('bios_type') is not None else None, console=otypes.Console(enabled=self.param('serial_console')) if self.param('serial_console') is not None else None, usb=otypes.Usb(enabled=self.param('usb_support')) if self.param('usb_support') is not None else None, sso=otypes.Sso(methods=[otypes.Method(id=otypes.SsoMethod.GUEST_AGENT)] if self.param('sso') else []) if self.param('sso') is not None else None, quota=otypes.Quota(id=self._module.params.get('quota_id')) if self.param('quota_id') is not None else None, high_availability=otypes.HighAvailability(enabled=self.param('high_availability'), priority=self.param('high_availability_priority')) if self.param('high_availability') is not None or self.param('high_availability_priority') else None, lease=otypes.StorageDomainLease(storage_domain=otypes.StorageDomain(id=get_id_by_name(service=self._connection.system_service().storage_domains_service(), name=self.param('lease')) if self.param('lease') else None)) if self.param('lease') is not None else None, cpu=otypes.Cpu(topology=otypes.CpuTopology(cores=self.param('cpu_cores'), sockets=self.param('cpu_sockets'), threads=self.param('cpu_threads')) if any((self.param('cpu_cores'), self.param('cpu_sockets'), self.param('cpu_threads'))) else None, cpu_tune=otypes.CpuTune(vcpu_pins=[otypes.VcpuPin(vcpu=int(pin['vcpu']), cpu_set=str(pin['cpu'])) for pin in self.param('cpu_pinning')]) if self.param('cpu_pinning') else None, mode=otypes.CpuMode(self.param('cpu_mode')) if self.param('cpu_mode') else None) if any((self.param('cpu_cores'), self.param('cpu_sockets'), self.param('cpu_threads'), self.param('cpu_mode'), self.param('cpu_pinning'))) else None, cpu_shares=self.param('cpu_shares'), virtio_scsi=otypes.VirtioScsi(enabled=self.param('virtio_scsi_enabled')) if self.param('virtio_scsi_enabled') is not None else None, multi_queues_enabled=self.param('multi_queues_enabled'), virtio_scsi_multi_queues=self.param('virtio_scsi_multi_queues'), tpm_enabled=self.param('tpm_enabled'), os=otypes.OperatingSystem(type=self.param('operating_system'), boot=otypes.Boot(devices=[otypes.BootDevice(dev) for dev in self.param('boot_devices')]) if self.param('boot_devices') else None, cmdline=self.param('kernel_params') if self.param('kernel_params_persist') else None, initrd=self.param('initrd_path') if self.param('kernel_params_persist') else None, kernel=self.param('kernel_path') if self.param('kernel_params_persist') else None) if self.param('operating_system') or self.param('boot_devices') or self.param('kernel_params_persist') else None, type=otypes.VmType(self.param('type')) if self.param('type') else None, memory=convert_to_bytes(self.param('memory')) if self.param('memory') else None, memory_policy=otypes.MemoryPolicy(guaranteed=convert_to_bytes(self.param('memory_guaranteed')), ballooning=self.param('ballooning_enabled'), max=convert_to_bytes(self.param('memory_max'))) if any((self.param('memory_guaranteed'), self.param('ballooning_enabled') is not None, self.param('memory_max'))) else None, instance_type=otypes.InstanceType(id=get_id_by_name(self._connection.system_service().instance_types_service(), self.param('instance_type'))) if self.param('instance_type') else None, custom_compatibility_version=otypes.Version(major=self._get_major(self.param('custom_compatibility_version')), minor=self._get_minor(self.param('custom_compatibility_version'))) if self.param('custom_compatibility_version') is not None else None, description=self.param('description'), comment=self.param('comment'), time_zone=otypes.TimeZone(name=self.param('timezone')) if self.param('timezone') else None, serial_number=otypes.SerialNumber(policy=otypes.SerialNumberPolicy(self.param('serial_policy')), value=self.param('serial_policy_value')) if self.param('serial_policy') is not None or self.param('serial_policy_value') is not None else None, placement_policy=placement_policy, soundcard_enabled=self.param('soundcard_enabled'), display=otypes.Display(smartcard_enabled=self.param('smartcard_enabled'), disconnect_action=display.get('disconnect_action'), keyboard_layout=display.get('keyboard_layout'), monitors=display.get('monitors'), copy_paste_enabled=display.get('copy_paste_enabled'), file_transfer_enabled=display.get('file_transfer_enabled')) if self.param('smartcard_enabled') is not None or display.get('copy_paste_enabled') is not None or display.get('file_transfer_enabled') is not None or (display.get('disconnect_action') is not None) or (display.get('keyboard_layout') is not None) or (display.get('monitors') is not None) else None, io=otypes.Io(threads=self.param('io_threads')) if self.param('io_threads') is not None else None, numa_tune_mode=otypes.NumaTuneMode(self.param('numa_tune_mode')) if self.param('numa_tune_mode') else None, storage_error_resume_behaviour=otypes.VmStorageErrorResumeBehaviour(self.param('storage_error_resume_behaviour')) if self.param('storage_error_resume_behaviour') else None, rng_device=otypes.RngDevice(source=otypes.RngSource(self.param('rng_device'))) if self.param('rng_device') else None, custom_properties=[otypes.CustomProperty(name=cp.get('name'), regexp=cp.get('regexp'), value=str(cp.get('value'))) for cp in self.param('custom_properties') if cp] if self.param('custom_properties') is not None else None, initialization=self.get_initialization() if self.param('cloud_init_persist') else None, snapshots=[otypes.Snapshot(id=snapshot.id)] if snapshot is not None else None)

    def _get_export_domain_service(self):
        provider_name = self._module.params['export_domain']
        export_sds_service = self._connection.system_service().storage_domains_service()
        export_sd_id = get_id_by_name(export_sds_service, provider_name)
        return export_sds_service.service(export_sd_id)

    def post_export_action(self, entity):
        self._service = self._get_export_domain_service().vms_service()

    def update_check(self, entity):
        res = self._update_check(entity)
        if entity.next_run_configuration_exists:
            res = res and self._update_check(self._service.service(entity.id).get(next_run=True))
        return res

    def _update_check(self, entity):

        def check_cpu_pinning():
            if self.param('cpu_pinning'):
                current = []
                if entity.cpu.cpu_tune:
                    current = [(str(pin.cpu_set), int(pin.vcpu)) for pin in entity.cpu.cpu_tune.vcpu_pins]
                passed = [(str(pin['cpu']), int(pin['vcpu'])) for pin in self.param('cpu_pinning')]
                return sorted(current) == sorted(passed)
            return True

        def check_custom_properties():
            if self.param('custom_properties'):
                current = []
                if entity.custom_properties:
                    current = [(cp.name, cp.regexp, str(cp.value)) for cp in entity.custom_properties]
                passed = [(cp.get('name'), cp.get('regexp'), str(cp.get('value'))) for cp in self.param('custom_properties') if cp]
                return sorted(current) == sorted(passed)
            return True

        def check_placement_policy():
            if self.param('placement_policy'):
                hosts = sorted(map(lambda host: self._connection.follow_link(host).name, entity.placement_policy.hosts if entity.placement_policy.hosts else []))
                if self.param('placement_policy_hosts'):
                    return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal(sorted(self.param('placement_policy_hosts')), hosts)
                return equal(self.param('placement_policy'), str(entity.placement_policy.affinity) if entity.placement_policy else None) and equal([self.param('host')], hosts)
            return True

        def check_host():
            if self.param('host') is not None:
                return self.param('host') in [self._connection.follow_link(host).name for host in getattr(entity.placement_policy, 'hosts', None) or []]
            return True

        def check_custom_compatibility_version():
            if self.param('custom_compatibility_version') is not None:
                return self._get_minor(self.param('custom_compatibility_version')) == self._get_minor(entity.custom_compatibility_version) and self._get_major(self.param('custom_compatibility_version')) == self._get_major(entity.custom_compatibility_version)
            return True
        cpu_mode = getattr(entity.cpu, 'mode')
        vm_display = entity.display
        provided_vm_display = self.param('graphical_console') or dict()
        return check_cpu_pinning() and check_custom_properties() and check_host() and check_placement_policy() and check_custom_compatibility_version() and (not self.param('cloud_init_persist')) and (not self.param('kernel_params_persist')) and equal(self.param('cluster'), get_link_name(self._connection, entity.cluster)) and equal(convert_to_bytes(self.param('memory')), entity.memory) and equal(convert_to_bytes(self.param('memory_guaranteed')), getattr(entity.memory_policy, 'guaranteed', None)) and equal(convert_to_bytes(self.param('memory_max')), getattr(entity.memory_policy, 'max', None)) and equal(self.param('cpu_cores'), getattr(getattr(entity.cpu, 'topology', None), 'cores', None)) and equal(self.param('cpu_sockets'), getattr(getattr(entity.cpu, 'topology', None), 'sockets', None)) and equal(self.param('cpu_threads'), getattr(getattr(entity.cpu, 'topology', None), 'threads', None)) and equal(self.param('cpu_mode'), str(cpu_mode) if cpu_mode else None) and equal(self.param('type'), str(entity.type)) and equal(self.param('name'), str(entity.name)) and equal(self.param('operating_system'), str(getattr(entity.os, 'type', None))) and equal(self.param('boot_menu'), getattr(getattr(entity.bios, 'boot_menu', None), 'enabled', None)) and equal(self.param('bios_type'), getattr(getattr(entity.bios, 'type', None), 'value', None)) and equal(self.param('soundcard_enabled'), entity.soundcard_enabled) and equal(self.param('smartcard_enabled'), getattr(vm_display, 'smartcard_enabled', False)) and equal(self.param('io_threads'), getattr(entity.io, 'threads', None)) and equal(self.param('ballooning_enabled'), getattr(entity.memory_policy, 'ballooning', None)) and equal(self.param('serial_console'), getattr(entity.console, 'enabled', None)) and equal(self.param('usb_support'), getattr(entity.usb, 'enabled', None)) and equal(self.param('sso'), True if getattr(entity.sso, 'methods', False) else False) and equal(self.param('quota_id'), getattr(entity.quota, 'id', None)) and equal(self.param('high_availability'), getattr(entity.high_availability, 'enabled', None)) and equal(self.param('high_availability_priority'), getattr(entity.high_availability, 'priority', None)) and equal(self.param('lease'), get_link_name(self._connection, getattr(entity.lease, 'storage_domain', None))) and equal(self.param('stateless'), entity.stateless) and equal(self.param('cpu_shares'), entity.cpu_shares) and equal(self.param('delete_protected'), entity.delete_protected) and equal(self.param('custom_emulated_machine'), entity.custom_emulated_machine) and equal(self.param('use_latest_template_version'), entity.use_latest_template_version) and equal(self.param('boot_devices'), [str(dev) for dev in getattr(getattr(entity.os, 'boot', None), 'devices', [])]) and equal(self.param('instance_type'), get_link_name(self._connection, entity.instance_type), ignore_case=True) and equal(self.param('description'), entity.description) and equal(self.param('comment'), entity.comment) and equal(self.param('timezone'), getattr(entity.time_zone, 'name', None)) and equal(self.param('serial_policy'), str(getattr(entity.serial_number, 'policy', None))) and equal(self.param('serial_policy_value'), getattr(entity.serial_number, 'value', None)) and equal(self.param('numa_tune_mode'), str(entity.numa_tune_mode)) and equal(self.param('storage_error_resume_behaviour'), str(entity.storage_error_resume_behaviour)) and equal(self.param('virtio_scsi_enabled'), getattr(entity.virtio_scsi, 'enabled', None)) and equal(self.param('multi_queues_enabled'), entity.multi_queues_enabled) and equal(self.param('virtio_scsi_multi_queues'), entity.virtio_scsi_multi_queues) and equal(self.param('tpm_enabled'), entity.tpm_enabled) and equal(self.param('rng_device'), str(entity.rng_device.source) if entity.rng_device else None) and equal(provided_vm_display.get('monitors'), getattr(vm_display, 'monitors', None)) and equal(provided_vm_display.get('copy_paste_enabled'), getattr(vm_display, 'copy_paste_enabled', None)) and equal(provided_vm_display.get('file_transfer_enabled'), getattr(vm_display, 'file_transfer_enabled', None)) and equal(provided_vm_display.get('keyboard_layout'), getattr(vm_display, 'keyboard_layout', None)) and equal(provided_vm_display.get('disconnect_action'), getattr(vm_display, 'disconnect_action', None), ignore_case=True)

    def pre_create(self, entity):
        if entity is None:
            self._is_new = True

    def post_update(self, entity):
        self.post_present(entity.id)

    def post_present(self, entity_id):
        entity = self._service.service(entity_id).get()
        self.__attach_disks(entity)
        self.__attach_nics(entity)
        self._attach_cd(entity)
        self.changed = self.__attach_numa_nodes(entity)
        self.changed = self.__attach_watchdog(entity)
        self.changed = self.__attach_graphical_console(entity)
        self.changed = self.__attach_host_devices(entity)
        self._wait_after_lease()

    def pre_remove(self, entity):
        if entity.status != otypes.VmStatus.DOWN:
            if not self._module.check_mode:
                self.changed = self.action(action='stop', action_condition=lambda vm: vm.status != otypes.VmStatus.DOWN, wait_condition=lambda vm: vm.status == otypes.VmStatus.DOWN)['changed']

    def _wait_after_lease(self):
        if self.param('lease') and self.param('wait_after_lease') != 0:
            time.sleep(self.param('wait_after_lease'))

    def __suspend_shutdown_common(self, vm_service):
        if vm_service.get().status in [otypes.VmStatus.MIGRATING, otypes.VmStatus.POWERING_UP, otypes.VmStatus.REBOOT_IN_PROGRESS, otypes.VmStatus.WAIT_FOR_LAUNCH, otypes.VmStatus.UP, otypes.VmStatus.RESTORING_STATE]:
            self._wait_for_UP(vm_service)

    def _pre_shutdown_action(self, entity):
        vm_service = self._service.vm_service(entity.id)
        self.__suspend_shutdown_common(vm_service)
        if entity.status in [otypes.VmStatus.SUSPENDED, otypes.VmStatus.PAUSED]:
            vm_service.start()
            self._wait_for_UP(vm_service)
        return vm_service.get()

    def _pre_suspend_action(self, entity):
        vm_service = self._service.vm_service(entity.id)
        self.__suspend_shutdown_common(vm_service)
        if entity.status in [otypes.VmStatus.PAUSED, otypes.VmStatus.DOWN]:
            vm_service.start()
            self._wait_for_UP(vm_service)
        return vm_service.get()

    def _post_start_action(self, entity):
        vm_service = self._service.service(entity.id)
        self._wait_for_UP(vm_service)
        self._attach_cd(vm_service.get())

    def __get_cds_from_sds(self, sds):
        for sd in sds:
            if sd.type == otypes.StorageDomainType.ISO:
                disks = sd.files
            elif sd.type == otypes.StorageDomainType.DATA:
                disks = sd.disks
            else:
                continue
            disks = list(filter(lambda x: (x.name == self.param('cd_iso') or x.id == self.param('cd_iso')) and (sd.type == otypes.StorageDomainType.ISO or x.content_type == otypes.DiskContentType.ISO), self._connection.follow_link(disks)))
            if disks:
                return disks

    def __get_cd_id(self):
        sds_service = self._connection.system_service().storage_domains_service()
        sds = sds_service.list(search='name="{0}"'.format(self.param('storage_domain') if self.param('storage_domain') else '*'))
        disks = self.__get_cds_from_sds(sds)
        if not disks:
            raise ValueError('Was not able to find disk with name or id "{0}".'.format(self.param('cd_iso')))
        if len(disks) > 1:
            raise ValueError('Found mutiple disks with same name "{0}" please use                 disk ID in "cd_iso" to specify which disk should be used.'.format(self.param('cd_iso')))
        return disks[0].id

    def _attach_cd(self, entity):
        cd_iso_id = self.param('cd_iso')
        if cd_iso_id is not None:
            if cd_iso_id:
                cd_iso_id = self.__get_cd_id()
            vm_service = self._service.service(entity.id)
            current = vm_service.get().status == otypes.VmStatus.UP and self.param('state') == 'running'
            cdroms_service = vm_service.cdroms_service()
            cdrom_device = cdroms_service.list()[0]
            cdrom_service = cdroms_service.cdrom_service(cdrom_device.id)
            cdrom = cdrom_service.get(current=current)
            if getattr(cdrom.file, 'id', '') != cd_iso_id:
                if not self._module.check_mode:
                    cdrom_service.update(cdrom=otypes.Cdrom(file=otypes.File(id=cd_iso_id)), current=current)
                self.changed = True
        return entity

    def _migrate_vm(self, entity):
        vm_host = self.param('host')
        vm_service = self._service.vm_service(entity.id)
        if entity.status == otypes.VmStatus.UP:
            if vm_host is not None:
                hosts_service = self._connection.system_service().hosts_service()
                clusters_service = self._connection.system_service().clusters_service()
                current_vm_host = hosts_service.host_service(entity.host.id).get().name
                if vm_host != current_vm_host:
                    if not self._module.check_mode:
                        vm_service.migrate(cluster=search_by_name(clusters_service, self.param('cluster')), host=otypes.Host(name=vm_host), force=self.param('force_migrate'))
                        self._wait_for_UP(vm_service)
                    self.changed = True
            elif self.param('migrate'):
                if not self._module.check_mode:
                    vm_service.migrate(force=self.param('force_migrate'))
                    self._wait_for_UP(vm_service)
                self.changed = True
        return entity

    def _wait_for_UP(self, vm_service):
        wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.UP, wait=self.param('wait'), timeout=self.param('timeout'))

    def _wait_for_vm_disks(self, vm_service):
        disks_service = self._connection.system_service().disks_service()
        for da in vm_service.disk_attachments_service().list():
            disk_service = disks_service.disk_service(da.disk.id)
            wait(service=disk_service, condition=lambda disk: disk.status == otypes.DiskStatus.OK if disk.storage_type == otypes.DiskStorageType.IMAGE else True, wait=self.param('wait'), timeout=self.param('timeout'))

    def wait_for_down(self, vm):
        """
        This function will first wait for the status DOWN of the VM.
        Then it will find the active snapshot and wait until it's state is OK for
        stateless VMs and stateless snapshot is removed.
        """
        vm_service = self._service.vm_service(vm.id)
        wait(service=vm_service, condition=lambda vm: vm.status == otypes.VmStatus.DOWN, wait=self.param('wait'), timeout=self.param('timeout'))
        if vm.stateless:
            snapshots_service = vm_service.snapshots_service()
            snapshots = snapshots_service.list()
            snap_active = [snap for snap in snapshots if snap.snapshot_type == otypes.SnapshotType.ACTIVE][0]
            snap_stateless = [snap for snap in snapshots if snap.snapshot_type == otypes.SnapshotType.STATELESS]
            if snap_stateless:
                "\n                We need to wait for Active snapshot ID, to be removed as it's current\n                stateless snapshot. Then we need to wait for staless snapshot ID to\n                be read, for use, because it will become active snapshot.\n                "
                wait(service=snapshots_service.snapshot_service(snap_active.id), condition=lambda snap: snap is None, wait=self.param('wait'), timeout=self.param('timeout'))
                wait(service=snapshots_service.snapshot_service(snap_stateless[0].id), condition=lambda snap: snap.snapshot_status == otypes.SnapshotStatus.OK, wait=self.param('wait'), timeout=self.param('timeout'))
        return True

    def __attach_graphical_console(self, entity):
        graphical_console = self.param('graphical_console')
        if not graphical_console:
            return False
        vm_service = self._service.service(entity.id)
        gcs_service = vm_service.graphics_consoles_service()
        graphical_consoles = gcs_service.list()
        if bool(graphical_console.get('headless_mode')):
            if not self._module.check_mode:
                for gc in graphical_consoles:
                    gcs_service.console_service(gc.id).remove()
            return len(graphical_consoles) > 0
        protocol = graphical_console.get('protocol')
        current_protocols = [str(gc.protocol) for gc in graphical_consoles]
        if not current_protocols:
            if not self._module.check_mode:
                for p in protocol:
                    gcs_service.add(otypes.GraphicsConsole(protocol=otypes.GraphicsType(p)))
            return True
        if protocol is not None and sorted(protocol) != sorted(current_protocols):
            if not self._module.check_mode:
                for gc in graphical_consoles:
                    gcs_service.console_service(gc.id).remove()
                for p in protocol:
                    gcs_service.add(otypes.GraphicsConsole(protocol=otypes.GraphicsType(p)))
            return True

    def __attach_disks(self, entity):
        if not self.param('disks'):
            return
        vm_service = self._service.service(entity.id)
        disks_service = self._connection.system_service().disks_service()
        disk_attachments_service = vm_service.disk_attachments_service()
        self._wait_for_vm_disks(vm_service)
        for disk in self.param('disks'):
            disk_id = disk.get('id')
            if disk_id is None:
                disk_id = getattr(search_by_name(service=disks_service, name=disk.get('name')), 'id', None)
            disk_attachment = disk_attachments_service.attachment_service(disk_id)
            if get_entity(disk_attachment) is None:
                if not self._module.check_mode:
                    disk_attachments_service.add(otypes.DiskAttachment(disk=otypes.Disk(id=disk_id), active=disk.get('activate', True), interface=otypes.DiskInterface(disk.get('interface', 'virtio')), bootable=disk.get('bootable', False)))
                self.changed = True

    def __get_vnic_profile_id(self, nic):
        """
        Return VNIC profile ID looked up by it's name, because there can be
        more VNIC profiles with same name, other criteria of filter is cluster.
        """
        vnics_service = self._connection.system_service().vnic_profiles_service()
        clusters_service = self._connection.system_service().clusters_service()
        cluster = search_by_name(clusters_service, self.param('cluster'))
        profiles = [profile for profile in vnics_service.list() if profile.name == nic.get('profile_name')]
        cluster_networks = [net.id for net in self._connection.follow_link(cluster.networks)]
        try:
            return next((profile.id for profile in profiles if profile.network.id in cluster_networks))
        except StopIteration:
            raise Exception("Profile '%s' was not found in cluster '%s'" % (nic.get('profile_name'), self.param('cluster')))

    def __get_numa_serialized(self, numa):
        return sorted([(x.index, [y.index for y in x.cpu.cores] if x.cpu else [], x.memory, [y.index for y in x.numa_node_pins] if x.numa_node_pins else []) for x in numa], key=lambda x: x[0])

    def __attach_numa_nodes(self, entity):
        numa_nodes_service = self._service.service(entity.id).numa_nodes_service()
        existed_numa_nodes = numa_nodes_service.list()
        if len(self.param('numa_nodes')) > 0:
            for current_numa_node in sorted(existed_numa_nodes, reverse=True, key=lambda x: x.index):
                numa_nodes_service.node_service(current_numa_node.id).remove()
        for numa_node in self.param('numa_nodes'):
            if numa_node is None or numa_node.get('index') is None or numa_node.get('cores') is None or (numa_node.get('memory') is None):
                continue
            numa_nodes_service.add(otypes.VirtualNumaNode(index=numa_node.get('index'), memory=numa_node.get('memory'), cpu=otypes.Cpu(cores=[otypes.Core(index=core) for core in numa_node.get('cores')]), numa_node_pins=[otypes.NumaNodePin(index=pin) for pin in numa_node.get('numa_node_pins')] if numa_node.get('numa_node_pins') is not None else None))
        return self.__get_numa_serialized(numa_nodes_service.list()) != self.__get_numa_serialized(existed_numa_nodes)

    def __attach_watchdog(self, entity):
        watchdogs_service = self._service.service(entity.id).watchdogs_service()
        watchdog = self.param('watchdog')
        if watchdog is not None:
            current_watchdog = next(iter(watchdogs_service.list()), None)
            if watchdog.get('model') is None and current_watchdog:
                watchdogs_service.watchdog_service(current_watchdog.id).remove()
                return True
            elif watchdog.get('model') is not None and current_watchdog is None:
                watchdogs_service.add(otypes.Watchdog(model=otypes.WatchdogModel(watchdog.get('model').lower()), action=otypes.WatchdogAction(watchdog.get('action'))))
                return True
            elif current_watchdog is not None:
                if str(current_watchdog.model).lower() != watchdog.get('model').lower() or str(current_watchdog.action).lower() != watchdog.get('action').lower():
                    watchdogs_service.watchdog_service(current_watchdog.id).update(otypes.Watchdog(model=otypes.WatchdogModel(watchdog.get('model')), action=otypes.WatchdogAction(watchdog.get('action'))))
                    return True
        return False

    def __attach_nics(self, entity):
        nics_service = self._service.service(entity.id).nics_service()
        for nic in self.param('nics'):
            if search_by_name(nics_service, nic.get('name')) is None:
                if not self._module.check_mode:
                    nics_service.add(otypes.Nic(name=nic.get('name'), interface=otypes.NicInterface(nic.get('interface', 'virtio')), vnic_profile=otypes.VnicProfile(id=self.__get_vnic_profile_id(nic)) if nic.get('profile_name') else None, mac=otypes.Mac(address=nic.get('mac_address')) if nic.get('mac_address') else None))
                self.changed = True

    def get_initialization(self):
        if self._initialization is not None:
            return self._initialization
        sysprep = self.param('sysprep')
        cloud_init = self.param('cloud_init')
        cloud_init_nics = self.param('cloud_init_nics') or []
        if cloud_init is not None:
            cloud_init_nics.append(cloud_init)
        if cloud_init or cloud_init_nics:
            self._initialization = otypes.Initialization(nic_configurations=[otypes.NicConfiguration(boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol').lower()) if nic.get('nic_boot_protocol') else None, ipv6_boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol_v6').lower()) if nic.get('nic_boot_protocol_v6') else None, name=nic.pop('nic_name', None), on_boot=True, ip=otypes.Ip(address=nic.pop('nic_ip_address', None), netmask=nic.pop('nic_netmask', None), gateway=nic.pop('nic_gateway', None), version=otypes.IpVersion('v4')) if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None else None, ipv6=otypes.Ip(address=nic.pop('nic_ip_address_v6', None), netmask=nic.pop('nic_netmask_v6', None), gateway=nic.pop('nic_gateway_v6', None), version=otypes.IpVersion('v6')) if nic.get('nic_gateway_v6') is not None or nic.get('nic_netmask_v6') is not None or nic.get('nic_ip_address_v6') is not None else None) for nic in cloud_init_nics if nic.get('nic_boot_protocol_v6') is not None or nic.get('nic_ip_address_v6') is not None or nic.get('nic_gateway_v6') is not None or (nic.get('nic_netmask_v6') is not None) or (nic.get('nic_gateway') is not None) or (nic.get('nic_netmask') is not None) or (nic.get('nic_ip_address') is not None) or (nic.get('nic_boot_protocol') is not None)] if cloud_init_nics else None, **cloud_init)
        elif sysprep:
            self._initialization = otypes.Initialization(**sysprep)
        return self._initialization

    def __attach_host_devices(self, entity):
        vm_service = self._service.service(entity.id)
        host_devices_service = vm_service.host_devices_service()
        host_devices = self.param('host_devices')
        updated = False
        if host_devices:
            device_names = [dev.name for dev in host_devices_service.list()]
            for device in host_devices:
                device_name = device.get('name')
                state = device.get('state', 'present')
                if state == 'absent' and device_name in device_names:
                    updated = True
                    if not self._module.check_mode:
                        device_id = get_id_by_name(host_devices_service, device.get('name'))
                        host_devices_service.device_service(device_id).remove()
                elif state == 'present' and device_name not in device_names:
                    updated = True
                    if not self._module.check_mode:
                        host_devices_service.add(otypes.HostDevice(name=device.get('name')))
        return updated