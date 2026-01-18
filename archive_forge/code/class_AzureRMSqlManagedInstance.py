from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMSqlManagedInstance(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), subnet_id=dict(type='str'), identity=dict(type='dict', options=identity_spec), sku=dict(type='dict', options=sku_spec), managed_instance_create_mode=dict(type='str'), administrator_login=dict(type='str'), administrator_login_password=dict(type='str', no_log=True), license_type=dict(type='str', choices=['LicenseIncluded', 'BasePrice']), v_cores=dict(type='int', choices=[8, 16, 24, 32, 40, 64, 80]), storage_size_in_gb=dict(type='int'), collation=dict(type='str'), dns_zone=dict(type='str'), dns_zone_partner=dict(type='str'), public_data_endpoint_enabled=dict(type='bool'), source_managed_instance_id=dict(type='str'), restore_point_in_time=dict(type='str'), proxy_override=dict(type='str', choices=['Proxy', 'Redirect', 'Default']), timezone_id=dict(type='str'), instance_pool_id=dict(type='str'), maintenance_configuration_id=dict(type='str'), private_endpoint_connections=dict(type='list', elements='str'), minimal_tls_version=dict(type='str', choices=['None', '1.0', '1.1', '1.2']), storage_account_type=dict(type='str'), zone_redundant=dict(type='bool'), primary_user_assigned_identity_id=dict(type='str'), key_id=dict(type='str'), administrators=dict(type='str'), state=dict(type='str', choices=['present', 'absent'], default='present'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        self.location = None
        self.state = None
        self.body = dict()
        super(AzureRMSqlManagedInstance, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        self.body['location'] = self.location
        sql_managed_instance = self.get()
        changed = False
        if self.state == 'present':
            if sql_managed_instance:
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                if not self.default_compare(modifiers, self.body, sql_managed_instance, '', self.results):
                    changed = True
                if changed:
                    if not self.check_mode:
                        sql_managed_instance = self.create_or_update(self.body)
            else:
                changed = True
                if not self.check_mode:
                    sql_managed_instance = self.create_or_update(self.body)
        else:
            changed = True
            if not self.check_mode:
                sql_managed_instance = self.delete_sql_managed_instance()
        self.results['changed'] = changed
        self.results['state'] = sql_managed_instance
        return self.results

    def get(self):
        try:
            response = self.sql_client.managed_instances.get(self.resource_group, self.name)
            return self.to_dict(response)
        except ResourceNotFoundError:
            pass

    def update_sql_managed_instance(self, parameters):
        try:
            response = self.sql_client.managed_instances.begin_update(resource_group_name=self.resource_group, managed_instance_name=self.name, parameters=parameters)
            try:
                response = self.sql_client.managed_instances.get(resource_group_name=self.resource_group, managed_instance_name=self.name)
            except ResourceNotFoundError:
                self.fail("The resource created failed, can't get the facts")
            return self.to_dict(response)
        except Exception as exc:
            self.fail('Error when updating SQL managed instance {0}: {1}'.format(self.name, exc.message))

    def create_or_update(self, parameters):
        try:
            response = self.sql_client.managed_instances.begin_create_or_update(resource_group_name=self.resource_group, managed_instance_name=self.name, parameters=parameters)
            try:
                response = self.sql_client.managed_instances.get(resource_group_name=self.resource_group, managed_instance_name=self.name)
            except ResourceNotFoundError:
                self.fail("The resource created failed, can't get the facts")
            return self.to_dict(response)
        except Exception as exc:
            self.fail('Error when creating SQL managed instance {0}: {1}'.format(self.name, exc))

    def delete_sql_managed_instance(self):
        try:
            response = self.sql_client.managed_instances.begin_delete(self.resource_group, self.name)
        except Exception as exc:
            self.fail('Error when deleting SQL managed instance {0}: {1}'.format(self.name, exc))

    def to_dict(self, item):
        if not item:
            return None
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'id': d.get('id', None), 'name': d.get('name', None), 'location': d.get('location', None), 'type': d.get('type', None), 'tags': d.get('tags', None), 'identity': {'user_assigned_identities': d.get('identity', {}).get('user_assigned_identities', None), 'principal_id': d.get('identity', {}).get('principal_id', None), 'type': d.get('identity', {}).get('type', None), 'tenant_id': d.get('identity', {}).get('tenant_id', None)}, 'sku': {'name': d.get('sku', {}).get('name', None), 'size': d.get('sku', {}).get('size', None), 'family': d.get('sku', {}).get('family', None), 'tier': d.get('sku', {}).get('tier', None), 'capacity': d.get('sku', {}).get('capacity', None)}, 'provisioning_state': d.get('provisioning_state', None), 'managed_instance_create_mode': d.get('managed_instance_create_mode', None), 'fully_qualified_domain_name': d.get('fully_qualified_domain_name', None), 'administrator_login': d.get('administrator_login', None), 'subnet_id': d.get('subnet_id', None), 'state': d.get('state', None), 'license_type': d.get('license_type', None), 'v_cores': d.get('v_cores', None), 'storage_size_in_gb': d.get('storage_size_in_gb', None), 'collation': d.get('collation', None), 'dns_zone': d.get('dns_zone', None), 'dns_zone_partner': d.get('dns_zone_partner', None), 'public_data_endpoint_enabled': d.get('public_data_endpoint_enabled', None), 'source_managed_instance_id': d.get('source_managed_instance_id', None), 'restore_point_in_time': d.get('restore_point_in_time', None), 'proxy_override': d.get('proxy_override', None), 'timezone_id': d.get('timezone_id', None), 'instance_pool_id': d.get('instance_pool_id', None), 'maintenance_configuration_id': d.get('maintenance_configuration_id', None), 'private_endpoint_connections': d.get('private_endpoint_connections', None), 'minimal_tls_version': d.get('minimal_tls_version', None), 'storage_account_type': d.get('storage_account_type', None), 'zone_redundant': d.get('zone_redundant', None), 'primary_user_assigned_identity_id': d.get('primary_user_assigned_identity_id', None), 'key_id': d.get('key_id', None), 'administrators': d.get('administrators', None)}
        return d