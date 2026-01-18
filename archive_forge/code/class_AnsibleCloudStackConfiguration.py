from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackConfiguration(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackConfiguration, self).__init__(module)
        self.returns = {'category': 'category', 'scope': 'scope', 'value': 'value'}
        self.storage = None
        self.account = None
        self.cluster = None

    def _get_common_configuration_args(self):
        args = {'name': self.module.params.get('name'), 'accountid': self.get_account(key='id'), 'storageid': self.get_storage(key='id'), 'zoneid': self.get_zone(key='id'), 'clusterid': self.get_cluster(key='id')}
        return args

    def get_zone(self, key=None):
        zone = self.module.params.get('zone')
        if zone:
            return super(AnsibleCloudStackConfiguration, self).get_zone(key=key)

    def get_cluster(self, key=None):
        if not self.cluster:
            cluster_name = self.module.params.get('cluster')
            if not cluster_name:
                return None
            args = {'name': cluster_name}
            clusters = self.query_api('listClusters', **args)
            if clusters:
                self.cluster = clusters['cluster'][0]
                self.result['cluster'] = self.cluster['name']
            else:
                self.module.fail_json(msg='Cluster %s not found.' % cluster_name)
        return self._get_by_key(key=key, my_dict=self.cluster)

    def get_storage(self, key=None):
        if not self.storage:
            storage_pool_name = self.module.params.get('storage')
            if not storage_pool_name:
                return None
            args = {'name': storage_pool_name}
            storage_pools = self.query_api('listStoragePools', **args)
            if storage_pools:
                self.storage = storage_pools['storagepool'][0]
                self.result['storage'] = self.storage['name']
            else:
                self.module.fail_json(msg='Storage pool %s not found.' % storage_pool_name)
        return self._get_by_key(key=key, my_dict=self.storage)

    def get_configuration(self):
        configuration = None
        args = self._get_common_configuration_args()
        args['fetch_list'] = True
        configurations = self.query_api('listConfigurations', **args)
        if not configurations:
            self.module.fail_json(msg='Configuration %s not found.' % args['name'])
        for config in configurations:
            if args['name'] == config['name']:
                configuration = config
        return configuration

    def get_value(self):
        value = str(self.module.params.get('value'))
        if value in ('True', 'False'):
            value = value.lower()
        return value

    def present_configuration(self):
        configuration = self.get_configuration()
        args = self._get_common_configuration_args()
        args['value'] = self.get_value()
        empty_value = args['value'] in [None, ''] and 'value' not in configuration
        if self.has_changed(args, configuration, ['value']) and (not empty_value):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateConfiguration', **args)
                configuration = res['configuration']
        return configuration

    def get_result(self, resource):
        self.result = super(AnsibleCloudStackConfiguration, self).get_result(resource)
        if self.account:
            self.result['account'] = self.account['name']
            self.result['domain'] = self.domain['path']
        elif self.zone:
            self.result['zone'] = self.zone['name']
        return self.result