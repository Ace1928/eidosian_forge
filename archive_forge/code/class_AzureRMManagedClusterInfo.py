from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMManagedClusterInfo(AzureRMModuleBase):
    """Utility class to get Azure Kubernetes Service facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'), show_kubeconfig=dict(type='str', choices=['user', 'admin']))
        self.results = dict(changed=False, aks=[], available_versions=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.show_kubeconfig = None
        super(AzureRMManagedClusterInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_aks_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_aks_facts' module has been renamed to 'azure_rm_aks_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        if self.name is not None and self.resource_group is not None:
            self.results['aks'] = self.get_item()
        elif self.resource_group is not None:
            self.results['aks'] = self.list_by_resourcegroup()
        else:
            self.results['aks'] = self.list_items()
        return self.results

    def get_item(self):
        """Get a single Azure Kubernetes Service"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.managedcluster_client.managed_clusters.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_obj(item, AZURE_OBJECT_CLASS)]
            if self.show_kubeconfig:
                result[0]['kube_config'] = self.get_aks_kubeconfig(self.resource_group, self.name)
        return result

    def list_by_resourcegroup(self):
        """Get all Azure Kubernetes Services"""
        self.log('List all Azure Kubernetes Services under resource group')
        try:
            response = self.managedcluster_client.managed_clusters.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                item_dict = self.serialize_obj(item, AZURE_OBJECT_CLASS)
                if self.show_kubeconfig:
                    item_dict['kube_config'] = self.get_aks_kubeconfig(self.resource_group, item.name)
                results.append(item_dict)
        return results

    def list_items(self):
        """Get all Azure Kubernetes Services"""
        self.log('List all Azure Kubernetes Services')
        try:
            response = self.managedcluster_client.managed_clusters.list()
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                item_dict = self.serialize_obj(item, AZURE_OBJECT_CLASS)
                if self.show_kubeconfig:
                    item_dict['kube_config'] = self.get_aks_kubeconfig(item.resource_group, item.name)
                results.append(item_dict)
        return results

    def get_aks_kubeconfig(self, resource_group, name):
        """
        Gets kubeconfig for the specified AKS instance.

        :return: AKS instance kubeconfig
        """
        if not self.show_kubeconfig:
            return ''
        role_name = 'cluster{0}'.format(str.capitalize(self.show_kubeconfig))
        access_profile = self.managedcluster_client.managed_clusters.get_access_profile(resource_group, name, role_name)
        return access_profile.kube_config.decode('utf-8')