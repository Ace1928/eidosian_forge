from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackStoragePool(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackStoragePool, self).__init__(module)
        self.returns = {'capacityiops': 'capacity_iops', 'podname': 'pod', 'clustername': 'cluster', 'disksizeallocated': 'disk_size_allocated', 'disksizetotal': 'disk_size_total', 'disksizeused': 'disk_size_used', 'scope': 'scope', 'hypervisor': 'hypervisor', 'type': 'type', 'ip_address': 'ipaddress', 'path': 'path', 'overprovisionfactor': 'overprovision_factor', 'storagecapabilities': 'storage_capabilities', 'suitableformigration': 'suitable_for_migration'}
        self.allocation_states = {'Up': 'enabled', 'Disabled': 'disabled', 'Maintenance': 'maintenance'}
        self.storage_pool = None

    def _get_common_args(self):
        return {'name': self.module.params.get('name'), 'url': self.module.params.get('storage_url'), 'zoneid': self.get_zone(key='id'), 'provider': self.get_storage_provider(), 'scope': self.module.params.get('scope'), 'hypervisor': self.module.params.get('hypervisor'), 'capacitybytes': self.module.params.get('capacity_bytes'), 'capacityiops': self.module.params.get('capacity_iops')}

    def _allocation_state_enabled_disabled_changed(self, pool, allocation_state):
        if allocation_state in ['enabled', 'disabled']:
            for pool_state, param_state in self.allocation_states.items():
                if pool_state == pool['state'] and allocation_state != param_state:
                    return True
        return False

    def _handle_allocation_state(self, pool, state=None):
        allocation_state = state or self.module.params.get('allocation_state')
        if not allocation_state:
            return pool
        if self.allocation_states.get(pool['state']) == allocation_state:
            return pool
        elif allocation_state in ['enabled', 'disabled']:
            pool = self._cancel_maintenance(pool)
            pool = self._update_storage_pool(pool=pool, allocation_state=allocation_state)
        elif allocation_state == 'maintenance':
            pool = self._update_storage_pool(pool=pool, allocation_state='enabled')
            pool = self._enable_maintenance(pool=pool)
        return pool

    def _create_storage_pool(self):
        args = self._get_common_args()
        args.update({'clusterid': self.get_cluster(key='id'), 'podid': self.get_pod(key='id'), 'managed': self.module.params.get('managed')})
        scope = self.module.params.get('scope')
        if scope is None:
            args['scope'] = 'cluster' if args['clusterid'] else 'zone'
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('createStoragePool', **args)
            return res['storagepool']

    def _update_storage_pool(self, pool, allocation_state=None):
        args = {'id': pool['id'], 'capacitybytes': self.module.params.get('capacity_bytes'), 'capacityiops': self.module.params.get('capacity_iops'), 'tags': self.get_storage_tags()}
        if self.has_changed(args, pool) or self._allocation_state_enabled_disabled_changed(pool, allocation_state):
            self.result['changed'] = True
            args['enabled'] = allocation_state == 'enabled' if allocation_state in ['enabled', 'disabled'] else None
            if not self.module.check_mode:
                res = self.query_api('updateStoragePool', **args)
                pool = res['storagepool']
        return pool

    def _enable_maintenance(self, pool):
        if pool['state'].lower() != 'maintenance':
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('enableStorageMaintenance', id=pool['id'])
                pool = self.poll_job(res, 'storagepool')
        return pool

    def _cancel_maintenance(self, pool):
        if pool['state'].lower() == 'maintenance':
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('cancelStorageMaintenance', id=pool['id'])
                pool = self.poll_job(res, 'storagepool')
        return pool

    def get_storage_tags(self):
        storage_tags = self.module.params.get('storage_tags')
        if storage_tags is None:
            return None
        return ','.join(storage_tags)

    def get_storage_pool(self, key=None):
        if self.storage_pool is None:
            zoneid = self.get_zone(key='id')
            clusterid = self.get_cluster(key='id')
            podid = self.get_pod(key='id')
            args = {'zoneid': zoneid, 'podid': podid, 'clusterid': clusterid, 'name': self.module.params.get('name')}
            res = self.query_api('listStoragePools', **args)
            if 'storagepool' not in res:
                return None
            self.storage_pool = res['storagepool'][0]
        return self.storage_pool

    def present_storage_pool(self):
        pool = self.get_storage_pool()
        if pool:
            pool = self._update_storage_pool(pool=pool)
        else:
            pool = self._create_storage_pool()
        if pool:
            pool = self._handle_allocation_state(pool=pool)
        return pool

    def absent_storage_pool(self):
        pool = self.get_storage_pool()
        if pool:
            self.result['changed'] = True
            args = {'id': pool['id']}
            if not self.module.check_mode:
                self._handle_allocation_state(pool=pool, state='maintenance')
                self.query_api('deleteStoragePool', **args)
        return pool

    def get_storage_provider(self, type='primary'):
        args = {'type': type}
        provider = self.module.params.get('provider')
        storage_providers = self.query_api('listStorageProviders', **args)
        for sp in storage_providers.get('dataStoreProvider') or []:
            if sp['name'].lower() == provider.lower():
                return provider
        self.fail_json(msg='Storage provider %s not found' % provider)

    def get_cluster(self, key=None):
        cluster = self.module.params.get('cluster')
        if not cluster:
            return None
        args = {'name': cluster, 'zoneid': self.get_zone(key='id')}
        clusters = self.query_api('listClusters', **args)
        if clusters:
            return self._get_by_key(key, clusters['cluster'][0])
        self.fail_json(msg='Cluster %s not found' % cluster)

    def get_result(self, resource):
        super(AnsibleCloudStackStoragePool, self).get_result(resource)
        if resource:
            self.result['storage_url'] = '%s://%s/%s' % (resource['type'], resource['ipaddress'], resource['path'])
            self.result['scope'] = resource['scope'].lower()
            self.result['storage_tags'] = resource['tags'].split(',') if resource.get('tags') else []
            self.result['allocation_state'] = self.allocation_states.get(resource['state'])
        return self.result