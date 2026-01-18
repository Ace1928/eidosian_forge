from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class ClustersModule(BaseModule):

    def __get_major(self, full_version):
        if full_version is None:
            return None
        if isinstance(full_version, otypes.Version):
            return full_version.major
        return int(full_version.split('.')[0])

    def __get_minor(self, full_version):
        if full_version is None:
            return None
        if isinstance(full_version, otypes.Version):
            return full_version.minor
        return int(full_version.split('.')[1])

    def param(self, name, default=None):
        return self._module.params.get(name, default)

    def _get_memory_policy(self):
        memory_policy = self.param('memory_policy')
        if memory_policy == 'desktop':
            return 200
        elif memory_policy == 'server':
            return 150
        elif memory_policy == 'disabled':
            return 100

    def _get_policy_id(self):
        migration_policy = self.param('migration_policy')
        if migration_policy == 'legacy':
            return '00000000-0000-0000-0000-000000000000'
        elif migration_policy == 'minimal_downtime':
            return '80554327-0569-496b-bdeb-fcbbf52b827b'
        elif migration_policy == 'suspend_workload':
            return '80554327-0569-496b-bdeb-fcbbf52b827c'
        elif migration_policy == 'post_copy':
            return 'a7aeedb2-8d66-4e51-bb22-32595027ce71'

    def _get_sched_policy(self):
        sched_policy = None
        if self.param('scheduling_policy'):
            sched_policies_service = self._connection.system_service().scheduling_policies_service()
            sched_policy = search_by_name(sched_policies_service, self.param('scheduling_policy'))
            if not sched_policy:
                raise Exception("Scheduling policy '%s' was not found" % self.param('scheduling_policy'))
        return sched_policy

    def _get_mac_pool(self):
        mac_pool = None
        if self._module.params.get('mac_pool'):
            mac_pool = search_by_name(self._connection.system_service().mac_pools_service(), self._module.params.get('mac_pool'))
        return mac_pool

    def _get_external_network_providers(self):
        return self.param('external_network_providers') or []

    def _get_external_network_provider_id(self, external_provider):
        return external_provider.get('id') or get_id_by_name(self._connection.system_service().openstack_network_providers_service(), external_provider.get('name'))

    def _get_external_network_providers_entity(self):
        if self.param('external_network_providers') is not None:
            return [otypes.ExternalProvider(id=self._get_external_network_provider_id(external_provider)) for external_provider in self.param('external_network_providers')]

    def build_entity(self):
        sched_policy = self._get_sched_policy()
        return otypes.Cluster(id=self.param('id'), name=self.param('name'), comment=self.param('comment'), description=self.param('description'), ballooning_enabled=self.param('ballooning'), gluster_service=self.param('gluster'), virt_service=self.param('virt'), threads_as_cores=self.param('threads_as_cores'), ha_reservation=self.param('ha_reservation'), trusted_service=self.param('trusted_service'), optional_reason=self.param('vm_reason'), maintenance_reason_required=self.param('host_reason'), scheduling_policy=otypes.SchedulingPolicy(id=sched_policy.id) if sched_policy else None, serial_number=otypes.SerialNumber(policy=otypes.SerialNumberPolicy(self.param('serial_policy')), value=self.param('serial_policy_value')) if self.param('serial_policy') is not None or self.param('serial_policy_value') is not None else None, migration=otypes.MigrationOptions(auto_converge=otypes.InheritableBoolean(self.param('migration_auto_converge')) if self.param('migration_auto_converge') else None, bandwidth=otypes.MigrationBandwidth(assignment_method=otypes.MigrationBandwidthAssignmentMethod(self.param('migration_bandwidth')) if self.param('migration_bandwidth') else None, custom_value=self.param('migration_bandwidth_limit')) if self.param('migration_bandwidth') or self.param('migration_bandwidth_limit') else None, compressed=otypes.InheritableBoolean(self.param('migration_compressed')) if self.param('migration_compressed') else None, encrypted=otypes.InheritableBoolean(self.param('migration_encrypted')) if self.param('migration_encrypted') else None, policy=otypes.MigrationPolicy(id=self._get_policy_id()) if self.param('migration_policy') else None) if self.param('migration_bandwidth') is not None or self.param('migration_bandwidth_limit') is not None or self.param('migration_auto_converge') is not None or (self.param('migration_compressed') is not None) or (self.param('migration_encrypted') is not None) or (self.param('migration_policy') is not None) else None, error_handling=otypes.ErrorHandling(on_error=otypes.MigrateOnError(self.param('resilience_policy'))) if self.param('resilience_policy') else None, fencing_policy=otypes.FencingPolicy(enabled=self.param('fence_enabled'), skip_if_gluster_bricks_up=self.param('fence_skip_if_gluster_bricks_up'), skip_if_gluster_quorum_not_met=self.param('fence_skip_if_gluster_quorum_not_met'), skip_if_connectivity_broken=otypes.SkipIfConnectivityBroken(enabled=self.param('fence_skip_if_connectivity_broken'), threshold=self.param('fence_connectivity_threshold')) if self.param('fence_skip_if_connectivity_broken') is not None or self.param('fence_connectivity_threshold') is not None else None, skip_if_sd_active=otypes.SkipIfSdActive(enabled=self.param('fence_skip_if_sd_active')) if self.param('fence_skip_if_sd_active') is not None else None) if self.param('fence_enabled') is not None or self.param('fence_skip_if_sd_active') is not None or self.param('fence_skip_if_connectivity_broken') is not None or (self.param('fence_skip_if_gluster_bricks_up') is not None) or (self.param('fence_skip_if_gluster_quorum_not_met') is not None) or (self.param('fence_connectivity_threshold') is not None) else None, display=otypes.Display(proxy=self.param('spice_proxy')) if self.param('spice_proxy') else None, required_rng_sources=[otypes.RngSource(rng) for rng in self.param('rng_sources')] if self.param('rng_sources') else None, memory_policy=otypes.MemoryPolicy(over_commit=otypes.MemoryOverCommit(percent=self._get_memory_policy())) if self.param('memory_policy') else None, ksm=otypes.Ksm(enabled=self.param('ksm'), merge_across_nodes=not self.param('ksm_numa')) if self.param('ksm_numa') is not None or self.param('ksm') is not None else None, data_center=otypes.DataCenter(name=self.param('data_center')) if self.param('data_center') else None, management_network=otypes.Network(name=self.param('network')) if self.param('network') else None, cpu=otypes.Cpu(architecture=otypes.Architecture(self.param('cpu_arch')) if self.param('cpu_arch') else None, type=self.param('cpu_type')) if self.param('cpu_arch') or self.param('cpu_type') else None, version=otypes.Version(major=self.__get_major(self.param('compatibility_version')), minor=self.__get_minor(self.param('compatibility_version'))) if self.param('compatibility_version') else None, switch_type=otypes.SwitchType(self.param('switch_type')) if self.param('switch_type') else None, mac_pool=otypes.MacPool(id=get_id_by_name(self._connection.system_service().mac_pools_service(), self.param('mac_pool'))) if self.param('mac_pool') else None, external_network_providers=self._get_external_network_providers_entity(), custom_scheduling_policy_properties=[otypes.Property(name=sp.get('name'), value=str(sp.get('value'))) for sp in self.param('scheduling_policy_properties') if sp] if self.param('scheduling_policy_properties') is not None else None, firewall_type=otypes.FirewallType(self.param('firewall_type')) if self.param('firewall_type') else None, gluster_tuned_profile=self.param('gluster_tuned_profile'))

    def _matches_entity(self, item, entity):
        return equal(item.get('id'), entity.id) and equal(item.get('name'), entity.name)

    def _update_check_external_network_providers(self, entity):
        if self.param('external_network_providers') is None:
            return True
        if entity.external_network_providers is None:
            return not self.param('external_network_providers')
        entity_providers = self._connection.follow_link(entity.external_network_providers)
        entity_provider_ids = [provider.id for provider in entity_providers]
        entity_provider_names = [provider.name for provider in entity_providers]
        for provider in self._get_external_network_providers():
            if provider.get('id'):
                if provider.get('id') not in entity_provider_ids:
                    return False
            elif provider.get('name') and provider.get('name') not in entity_provider_names:
                return False
        for entity_provider in entity_providers:
            if not any((self._matches_entity(provider, entity_provider) for provider in self._get_external_network_providers())):
                return False
        return True

    def update_check(self, entity):
        sched_policy = self._get_sched_policy()
        migration_policy = getattr(entity.migration, 'policy', None)
        cluster_cpu = getattr(entity, 'cpu', dict())

        def check_custom_scheduling_policy_properties():
            if self.param('scheduling_policy_properties'):
                current = []
                if entity.custom_scheduling_policy_properties:
                    current = [(sp.name, str(sp.value)) for sp in entity.custom_scheduling_policy_properties]
                passed = [(sp.get('name'), str(sp.get('value'))) for sp in self.param('scheduling_policy_properties') if sp]
                for p in passed:
                    if p not in current:
                        return False
            return True
        return check_custom_scheduling_policy_properties() and equal(self.param('name'), entity.name) and equal(self.param('comment'), entity.comment) and equal(self.param('description'), entity.description) and equal(self.param('switch_type'), str(entity.switch_type)) and equal(self.param('cpu_arch'), str(getattr(cluster_cpu, 'architecture', None))) and equal(self.param('cpu_type'), getattr(cluster_cpu, 'type', None)) and equal(self.param('ballooning'), entity.ballooning_enabled) and equal(self.param('gluster'), entity.gluster_service) and equal(self.param('virt'), entity.virt_service) and equal(self.param('threads_as_cores'), entity.threads_as_cores) and equal(self.param('ksm_numa'), not entity.ksm.merge_across_nodes) and equal(self.param('ksm'), entity.ksm.enabled) and equal(self.param('ha_reservation'), entity.ha_reservation) and equal(self.param('trusted_service'), entity.trusted_service) and equal(self.param('host_reason'), entity.maintenance_reason_required) and equal(self.param('vm_reason'), entity.optional_reason) and equal(self.param('spice_proxy'), getattr(entity.display, 'proxy', None)) and equal(self.param('fence_enabled'), entity.fencing_policy.enabled) and equal(self.param('fence_skip_if_gluster_bricks_up'), entity.fencing_policy.skip_if_gluster_bricks_up) and equal(self.param('fence_skip_if_gluster_quorum_not_met'), entity.fencing_policy.skip_if_gluster_quorum_not_met) and equal(self.param('fence_skip_if_sd_active'), entity.fencing_policy.skip_if_sd_active.enabled) and equal(self.param('fence_skip_if_connectivity_broken'), entity.fencing_policy.skip_if_connectivity_broken.enabled) and equal(self.param('fence_connectivity_threshold'), entity.fencing_policy.skip_if_connectivity_broken.threshold) and equal(self.param('resilience_policy'), str(entity.error_handling.on_error)) and equal(self.param('migration_bandwidth'), str(entity.migration.bandwidth.assignment_method)) and equal(self.param('migration_auto_converge'), str(entity.migration.auto_converge)) and equal(self.param('migration_compressed'), str(entity.migration.compressed)) and equal(self.param('migration_encrypted'), str(entity.migration.encrypted)) and equal(self.param('serial_policy'), str(getattr(entity.serial_number, 'policy', None))) and equal(self.param('serial_policy_value'), getattr(entity.serial_number, 'value', None)) and equal(self.param('scheduling_policy'), getattr(self._connection.follow_link(entity.scheduling_policy), 'name', None)) and equal(self.param('firewall_type'), str(entity.firewall_type)) and equal(self.param('gluster_tuned_profile'), getattr(entity, 'gluster_tuned_profile', None)) and equal(self._get_policy_id(), getattr(migration_policy, 'id', None)) and equal(self._get_memory_policy(), entity.memory_policy.over_commit.percent) and equal(self.__get_minor(self.param('compatibility_version')), self.__get_minor(entity.version)) and equal(self.__get_major(self.param('compatibility_version')), self.__get_major(entity.version)) and equal(self.param('migration_bandwidth_limit') if self.param('migration_bandwidth') == 'custom' else None, entity.migration.bandwidth.custom_value) and equal(sorted(self.param('rng_sources')) if self.param('rng_sources') else None, sorted([str(source) for source in entity.required_rng_sources])) and equal(get_id_by_name(self._connection.system_service().mac_pools_service(), self.param('mac_pool'), raise_error=False), entity.mac_pool.id) and self._update_check_external_network_providers(entity)