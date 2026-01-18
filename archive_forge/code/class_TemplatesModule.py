from __future__ import (absolute_import, division, print_function)
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class TemplatesModule(BaseModule):

    def __init__(self, *args, **kwargs):
        super(TemplatesModule, self).__init__(*args, **kwargs)
        self._initialization = None

    def build_entity(self):
        return otypes.Template(id=self._module.params['id'], name=self._module.params['name'], cluster=otypes.Cluster(name=self._module.params['cluster']) if self._module.params['cluster'] else None, vm=otypes.Vm(name=self._module.params['vm']) if self._module.params['vm'] else None, bios=otypes.Bios(boot_menu=otypes.BootMenu(enabled=self.param('boot_menu')) if self.param('boot_menu') is not None else None, type=otypes.BiosType[self.param('bios_type').upper()] if self.param('bios_type') is not None else None) if self.param('boot_menu') is not None or self.param('bios_type') is not None else None, description=self._module.params['description'], cpu_profile=otypes.CpuProfile(id=search_by_name(self._connection.system_service().cpu_profiles_service(), self._module.params['cpu_profile']).id) if self._module.params['cpu_profile'] else None, display=otypes.Display(smartcard_enabled=self.param('smartcard_enabled')) if self.param('smartcard_enabled') is not None else None, os=otypes.OperatingSystem(type=self.param('operating_system')) if self.param('operating_system') else None, memory=convert_to_bytes(self.param('memory')) if self.param('memory') else None, soundcard_enabled=self.param('soundcard_enabled'), usb=otypes.Usb(enabled=self.param('usb_support')) if self.param('usb_support') is not None else None, sso=otypes.Sso(methods=[otypes.Method(id=otypes.SsoMethod.GUEST_AGENT)] if self.param('sso') else []) if self.param('sso') is not None else None, time_zone=otypes.TimeZone(name=self.param('timezone')) if self.param('timezone') else None, version=otypes.TemplateVersion(base_template=self._get_base_template(), version_name=self.param('version').get('name')) if self.param('version') else None, memory_policy=otypes.MemoryPolicy(guaranteed=convert_to_bytes(self.param('memory_guaranteed')), ballooning=self.param('ballooning_enabled'), max=convert_to_bytes(self.param('memory_max'))) if any((self.param('memory_guaranteed'), self.param('ballooning_enabled'), self.param('memory_max'))) else None, io=otypes.Io(threads=self.param('io_threads')) if self.param('io_threads') is not None else None, initialization=self.get_initialization())

    def _get_base_template(self):
        templates = self._connection.system_service().templates_service().list()
        if not templates:
            return None
        template_name = self.param('name')
        named_templates = [t for t in templates if t.name == template_name]
        if not named_templates:
            return None
        base_template = min(named_templates, key=lambda x: x.version.version_number)
        return otypes.Template(id=base_template.id)

    def post_update(self, entity):
        self.post_present(entity.id)

    def post_present(self, entity_id):
        entity = self._service.service(entity_id).get()
        self.__attach_nics(entity)

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
            self._initialization = otypes.Initialization(nic_configurations=[otypes.NicConfiguration(boot_protocol=otypes.BootProtocol(nic.pop('nic_boot_protocol').lower()) if nic.get('nic_boot_protocol') else None, name=nic.pop('nic_name', None), on_boot=True, ip=otypes.Ip(address=nic.pop('nic_ip_address', None), netmask=nic.pop('nic_netmask', None), gateway=nic.pop('nic_gateway', None)) if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None else None) for nic in cloud_init_nics if nic.get('nic_gateway') is not None or nic.get('nic_netmask') is not None or nic.get('nic_ip_address') is not None or (nic.get('nic_boot_protocol') is not None)] if cloud_init_nics else None, **cloud_init)
        elif sysprep:
            self._initialization = otypes.Initialization(**sysprep)
        return self._initialization

    def update_check(self, entity):
        template_display = entity.display
        return equal(self._module.params.get('cluster'), get_link_name(self._connection, entity.cluster)) and equal(self._module.params.get('description'), entity.description) and equal(self.param('operating_system'), str(entity.os.type)) and equal(self.param('name'), str(entity.name)) and equal(self.param('smartcard_enabled'), getattr(template_display, 'smartcard_enabled', False)) and equal(self.param('soundcard_enabled'), entity.soundcard_enabled) and equal(self.param('ballooning_enabled'), entity.memory_policy.ballooning) and equal(self.param('boot_menu'), entity.bios.boot_menu.enabled) and equal(self.param('bios_type'), entity.bios.type.value) and equal(self.param('sso'), True if entity.sso.methods else False) and equal(self.param('timezone'), getattr(entity.time_zone, 'name', None)) and equal(self.param('usb_support'), entity.usb.enabled) and equal(convert_to_bytes(self.param('memory_guaranteed')), entity.memory_policy.guaranteed) and equal(convert_to_bytes(self.param('memory_max')), entity.memory_policy.max) and equal(convert_to_bytes(self.param('memory')), entity.memory) and equal(self._module.params.get('cpu_profile'), get_link_name(self._connection, entity.cpu_profile)) and equal(self.param('io_threads'), entity.io.threads)

    def _get_export_domain_service(self):
        provider_name = self._module.params['export_domain'] or self._module.params['image_provider']
        export_sds_service = self._connection.system_service().storage_domains_service()
        export_sd = search_by_name(export_sds_service, provider_name)
        if export_sd is None:
            raise ValueError("Export storage domain/Image Provider '%s' wasn't found." % provider_name)
        return export_sds_service.service(export_sd.id)

    def post_export_action(self, entity):
        self._service = self._get_export_domain_service().templates_service()

    def post_import_action(self, entity):
        self._service = self._connection.system_service().templates_service()