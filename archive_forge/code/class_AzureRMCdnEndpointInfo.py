from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
class AzureRMCdnEndpointInfo(AzureRMModuleBase):
    """Utility class to get Azure Azure CDN endpoint facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str', required=True), profile_name=dict(type='str', required=True), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, cdnendpoints=[])
        self.name = None
        self.resource_group = None
        self.profile_name = None
        self.tags = None
        super(AzureRMCdnEndpointInfo, self).__init__(supports_check_mode=True, derived_arg_spec=self.module_args, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_cdnendpoint_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_cdnendpoint_facts' module has been renamed to 'azure_rm_cdnendpoint_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.cdn_client = self.get_mgmt_svc_client(CdnManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2017-04-02')
        if self.name:
            self.results['cdnendpoints'] = self.get_item()
        else:
            self.results['cdnendpoints'] = self.list_by_profile()
        return self.results

    def get_item(self):
        """Get a single Azure Azure CDN endpoint"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.cdn_client.endpoints.get(self.resource_group, self.profile_name, self.name)
        except Exception:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_cdnendpoint(item)]
        return result

    def list_by_profile(self):
        """Get all Azure Azure CDN endpoints within an Azure CDN profile"""
        self.log('List all Azure CDN endpoints within an Azure CDN profile')
        try:
            response = self.cdn_client.endpoints.list_by_profile(self.resource_group, self.profile_name)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_cdnendpoint(item))
        return results

    def serialize_cdnendpoint(self, cdnendpoint):
        """
        Convert a Azure CDN endpoint object to dict.
        :param cdn: Azure CDN endpoint object
        :return: dict
        """
        result = self.serialize_obj(cdnendpoint, AZURE_OBJECT_CLASS)
        new_result = {}
        new_result['id'] = cdnendpoint.id
        new_result['resource_group'] = re.sub('\\/.*', '', re.sub('.*resourcegroups\\/', '', result['id']))
        new_result['profile_name'] = re.sub('\\/.*', '', re.sub('.*profiles\\/', '', result['id']))
        new_result['name'] = cdnendpoint.name
        new_result['type'] = cdnendpoint.type
        new_result['location'] = cdnendpoint.location
        new_result['resource_state'] = cdnendpoint.resource_state
        new_result['provisioning_state'] = cdnendpoint.provisioning_state
        new_result['query_string_caching_behavior'] = cdnendpoint.query_string_caching_behavior
        new_result['is_compression_enabled'] = cdnendpoint.is_compression_enabled
        new_result['is_http_allowed'] = cdnendpoint.is_http_allowed
        new_result['is_https_allowed'] = cdnendpoint.is_https_allowed
        new_result['content_types_to_compress'] = cdnendpoint.content_types_to_compress
        new_result['origin_host_header'] = cdnendpoint.origin_host_header
        new_result['origin_path'] = cdnendpoint.origin_path
        new_result['origin'] = dict(name=cdnendpoint.origins[0].name, host_name=cdnendpoint.origins[0].host_name, http_port=cdnendpoint.origins[0].http_port, https_port=cdnendpoint.origins[0].https_port)
        new_result['tags'] = cdnendpoint.tags
        return new_result