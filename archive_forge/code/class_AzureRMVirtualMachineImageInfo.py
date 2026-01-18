from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualMachineImageInfo(AzureRMModuleBase):

    def __init__(self, **kwargs):
        self.module_arg_spec = dict(location=dict(type='str', required=True), publisher=dict(type='str'), offer=dict(type='str'), sku=dict(type='str'), version=dict(type='str'))
        self.results = dict(changed=False)
        self.location = None
        self.publisher = None
        self.offer = None
        self.sku = None
        self.version = None
        super(AzureRMVirtualMachineImageInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachineimage_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachineimage_facts' module has been renamed to 'azure_rm_virtualmachineimage_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if is_old_facts:
            self.results['ansible_facts'] = dict()
            if self.location and self.publisher and self.offer and self.sku and self.version:
                self.results['ansible_facts']['azure_vmimages'] = self.get_item()
            elif self.location and self.publisher and self.offer and self.sku:
                self.results['ansible_facts']['azure_vmimages'] = self.list_images()
            elif self.location and self.publisher:
                self.results['ansible_facts']['azure_vmimages'] = self.list_offers()
            elif self.location:
                self.results['ansible_facts']['azure_vmimages'] = self.list_publishers()
        elif self.location and self.publisher and self.offer and self.sku and self.version:
            self.results['vmimages'] = self.get_item()
        elif self.location and self.publisher and self.offer and self.sku:
            self.results['vmimages'] = self.list_images()
        elif self.location and self.publisher:
            self.results['vmimages'] = self.list_offers()
        elif self.location:
            self.results['vmimages'] = self.list_publishers()
        return self.results

    def get_item(self):
        item = None
        result = []
        versions = None
        try:
            versions = self.compute_client.virtual_machine_images.list(self.location, self.publisher, self.offer, self.sku, top=1, orderby='name desc')
        except ResourceNotFoundError:
            pass
        if self.version == 'latest':
            item = versions[-1]
        else:
            for version in versions:
                if version.name == self.version:
                    item = version
        if item:
            result = [self.serialize_obj(item, 'VirtualMachineImage', enum_modules=AZURE_ENUM_MODULES)]
        return result

    def list_images(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_images.list(self.location, self.publisher, self.offer, self.sku)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list images: {0}'.format(str(exc)))
        if response:
            for item in response:
                results.append(self.serialize_obj(item, 'VirtualMachineImageResource', enum_modules=AZURE_ENUM_MODULES))
        return results

    def list_offers(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_images.list_offers(self.location, self.publisher)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list offers: {0}'.format(str(exc)))
        if response:
            for item in response:
                results.append(self.serialize_obj(item, 'VirtualMachineImageResource', enum_modules=AZURE_ENUM_MODULES))
        return results

    def list_publishers(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_images.list_publishers(self.location)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list publishers: {0}'.format(str(exc)))
        if response:
            for item in response:
                results.append(self.serialize_obj(item, 'VirtualMachineImageResource', enum_modules=AZURE_ENUM_MODULES))
        return results