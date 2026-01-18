from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
class AzureRMLockInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), managed_resource_id=dict(type='str'))
        self.results = dict(changed=False, locks=[])
        mutually_exclusive = [['resource_group', 'managed_resource_id']]
        self.name = None
        self.resource_group = None
        self.managed_resource_id = None
        self._mgmt_client = None
        self._query_parameters = {'api-version': '2016-09-01'}
        self._header_parameters = {'Content-Type': 'application/json; charset=utf-8'}
        super(AzureRMLockInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, facts_module=True, mutually_exclusive=mutually_exclusive, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_lock_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_lock_facts' module has been renamed to 'azure_rm_lock_info'", version=(2.9,))
        for key in self.module_arg_spec.keys():
            setattr(self, key, kwargs[key])
        self._mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        changed = False
        scope = self.get_scope()
        url = '/{0}/providers/Microsoft.Authorization/locks'.format(scope)
        if self.name:
            url = '{0}/{1}'.format(url, self.name)
        locks = self.list_locks(url)
        resp = locks.get('value') if 'value' in locks else [locks]
        self.results['locks'] = [self.to_dict(x) for x in resp]
        return self.results

    def to_dict(self, lock):
        resp = dict(id=lock['id'], name=lock['name'], level=_camel_to_snake(lock['properties']['level']), managed_resource_id=re.sub('/providers/Microsoft.Authorization/locks/.+', '', lock['id']))
        if lock['properties'].get('notes'):
            resp['notes'] = lock['properties']['notes']
        if lock['properties'].get('owners'):
            resp['owners'] = [x['application_id'] for x in lock['properties']['owners']]
        return resp

    def list_locks(self, url):
        try:
            resp = self._mgmt_client.query(url=url, method='GET', query_parameters=self._query_parameters, header_parameters=self._header_parameters, body=None, expected_status_codes=[200], polling_timeout=None, polling_interval=None)
            return json.loads(resp.body())
        except Exception as exc:
            self.fail('Error when finding locks {0}: {1}'.format(url, exc.message))

    def get_scope(self):
        """
        Get the resource scope of the lock management.
        '/subscriptions/{subscriptionId}' for subscriptions,
        '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}' for resource groups,
        '/subscriptions/{subscriptionId}/resourcegroups/{resourceGroupName}/providers/{namespace}/{resourceType}/{resourceName}' for resources.
        """
        if self.managed_resource_id:
            return self.managed_resource_id
        elif self.resource_group:
            return '/subscriptions/{0}/resourcegroups/{1}'.format(self.subscription_id, self.resource_group)
        else:
            return '/subscriptions/{0}'.format(self.subscription_id)