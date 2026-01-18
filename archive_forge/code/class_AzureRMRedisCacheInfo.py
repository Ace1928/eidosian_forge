from __future__ import absolute_import, division, print_function
import re
class AzureRMRedisCacheInfo(AzureRMModuleBase):
    """Utility class to get Azure Cache for Redis facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str', required=True), return_access_keys=dict(type='bool', default=False), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, rediscaches=[])
        self.name = None
        self.resource_group = None
        self.profile_name = None
        self.tags = None
        self._client = None
        super(AzureRMRedisCacheInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_rediscache_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_rediscache_facts' module has been renamed to 'azure_rm_rediscache_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self._client = self.get_mgmt_svc_client(RedisManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2018-03-01', is_track2=True)
        if self.name:
            self.results['rediscaches'] = self.get_item()
        else:
            self.results['rediscaches'] = self.list_by_resourcegroup()
        return self.results

    def get_item(self):
        """Get a single Azure Cache for Redis"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self._client.redis.get(resource_group_name=self.resource_group, name=self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_rediscache(item)]
        return result

    def list_by_resourcegroup(self):
        """Get all Azure Cache for Redis within a resource group"""
        self.log('List all Azure Cache for Redis within a resource group')
        try:
            response = self._client.redis.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_rediscache(item))
        return results

    def list_keys(self):
        """List Azure Cache for Redis keys"""
        self.log('List keys for {0}'.format(self.name))
        item = None
        try:
            item = self._client.redis.list_keys(resource_group_name=self.resource_group, name=self.name)
        except Exception as exc:
            self.fail('Failed to list redis keys of {0} - {1}'.format(self.name, str(exc)))
        return item

    def serialize_rediscache(self, rediscache):
        """
        Convert an Azure Cache for Redis object to dict.
        :param rediscache: Azure Cache for Redis object
        :return: dict
        """
        new_result = dict(id=rediscache.id, resource_group=re.sub('\\/.*', '', re.sub('.*resourceGroups\\/', '', rediscache.id)), name=rediscache.name, location=rediscache.location, provisioning_state=rediscache.provisioning_state, configuration=rediscache.redis_configuration, tenant_settings=rediscache.tenant_settings, minimum_tls_version=rediscache.minimum_tls_version, public_network_access=rediscache.public_network_access, redis_version=rediscache.redis_version, shard_count=rediscache.shard_count, enable_non_ssl_port=rediscache.enable_non_ssl_port, static_ip=rediscache.static_ip, subnet=rediscache.subnet_id, host_name=rediscache.host_name, tags=rediscache.tags)
        if rediscache.sku:
            new_result['sku'] = dict(name=rediscache.sku.name.lower(), size=rediscache.sku.family + str(rediscache.sku.capacity))
        if self.return_access_keys:
            access_keys = self.list_keys()
            if access_keys:
                new_result['access_keys'] = dict(primary=access_keys.primary_key, secondary=access_keys.secondary_key)
        return new_result