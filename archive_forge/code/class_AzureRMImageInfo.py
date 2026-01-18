from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMImageInfo(AzureRMModuleBase):

    def __init__(self, **kwargs):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.name = None
        self.format = None
        self.tags = None
        super(AzureRMImageInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_image_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_image_facts' module has been renamed to 'azure_rm_image_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and self.resource_group:
            self.results['images'] = self.get_image(self.resource_group, self.name)
        elif self.name and (not self.resource_group):
            self.results['images'] = self.list_images(self.name)
        elif not self.name and self.resource_group:
            self.results['images'] = self.list_images_by_resource_group(self.resource_group)
        elif not self.name and (not self.resource_group):
            self.results['images'] = self.list_images()
        return self.results

    def get_image(self, resource_group, image_name):
        """
        Returns image details based on its name
        """
        self.log('Get properties for {0}'.format(self.name))
        result = []
        item = None
        try:
            item = self.image_client.images.get(resource_group, image_name)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list images - {0}'.format(str(exc)))
        result = [self.format_item(item)]
        return result

    def list_images_by_resource_group(self, resource_group):
        """
        Returns image details based on its resource group
        """
        self.log('List images filtered by resource group')
        response = None
        try:
            response = self.image_client.images.list_by_resource_group(resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list images: {0}'.format(str(exc)))
        return [self.format_item(x) for x in response if self.has_tags(x.tags, self.tags)] if response else []

    def list_images(self, image_name=None):
        """
        Returns image details in current subscription
        """
        self.log('List images within current subscription')
        response = None
        results = []
        try:
            response = self.image_client.images.list()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all images: {0}'.format(str(exc)))
        results = [self.format_item(x) for x in response if self.has_tags(x.tags, self.tags)] if response else []
        if image_name:
            results = [result for result in results if result['name'] == image_name]
        return results

    def format_item(self, item):
        d = item.as_dict()
        for data_disk in d['storage_profile']['data_disks']:
            if 'managed_disk' in data_disk.keys():
                data_disk['managed_disk_id'] = data_disk['managed_disk']['id']
                data_disk.pop('managed_disk', None)
        d = {'id': d['id'], 'resource_group': d['id'].split('/')[4], 'name': d['name'], 'location': d['location'], 'tags': d.get('tags'), 'source': d['source_virtual_machine']['id'] if 'source_virtual_machine' in d.keys() else None, 'os_type': d['storage_profile']['os_disk']['os_type'], 'os_state': d['storage_profile']['os_disk']['os_state'], 'os_disk_caching': d['storage_profile']['os_disk']['caching'], 'os_storage_account_type': d['storage_profile']['os_disk']['storage_account_type'], 'os_disk': d['storage_profile']['os_disk']['managed_disk']['id'] if 'managed_disk' in d['storage_profile']['os_disk'].keys() else None, 'os_blob_uri': d['storage_profile']['os_disk']['blob_uri'] if 'blob_uri' in d['storage_profile']['os_disk'].keys() else None, 'provisioning_state': d['provisioning_state'], 'data_disks': d['storage_profile']['data_disks'], 'hyper_v_generation': d.get('hyper_v_generation')}
        return d