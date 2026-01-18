from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
class AzureRMCdnprofileInfo(AzureRMModuleBase):
    """Utility class to get Azure CDN profile facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, cdnprofiles=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.cdn_client = None
        super(AzureRMCdnprofileInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_cdnprofile_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_cdnprofile_facts' module has been renamed to 'azure_rm_cdnprofile_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.cdn_client = self.get_cdn_client()
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['cdnprofiles'] = self.get_item()
        elif self.resource_group:
            self.results['cdnprofiles'] = self.list_resource_group()
        else:
            self.results['cdnprofiles'] = self.list_all()
        return self.results

    def get_item(self):
        """Get a single Azure CDN profile"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.cdn_client.profiles.get(self.resource_group, self.name)
        except Exception:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_cdnprofile(item)]
        return result

    def list_resource_group(self):
        """Get all Azure CDN profiles within a resource group"""
        self.log('List all Azure CDNs within a resource group')
        try:
            response = self.cdn_client.profiles.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_cdnprofile(item))
        return results

    def list_all(self):
        """Get all Azure CDN profiles within a subscription"""
        self.log('List all CDN profiles within a subscription')
        try:
            response = self.cdn_client.profiles.list()
        except Exception as exc:
            self.fail('Error listing all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_cdnprofile(item))
        return results

    def serialize_cdnprofile(self, cdnprofile):
        """
        Convert a CDN profile object to dict.
        :param cdn: CDN profile object
        :return: dict
        """
        result = self.serialize_obj(cdnprofile, AZURE_OBJECT_CLASS)
        new_result = {}
        new_result['id'] = cdnprofile.id
        new_result['resource_group'] = re.sub('\\/.*', '', re.sub('.*resourcegroups\\/', '', result['id']))
        new_result['name'] = cdnprofile.name
        new_result['type'] = cdnprofile.type
        new_result['location'] = cdnprofile.location
        new_result['resource_state'] = cdnprofile.resource_state
        new_result['sku'] = cdnprofile.sku.name
        new_result['provisioning_state'] = cdnprofile.provisioning_state
        new_result['tags'] = cdnprofile.tags
        return new_result

    def get_cdn_client(self):
        if not self.cdn_client:
            self.cdn_client = self.get_mgmt_svc_client(CdnManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2017-04-02')
        return self.cdn_client