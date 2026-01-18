from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAKSUpgrade(AzureRMModuleBase):
    """
    Utility class to get Azure Kubernetes Service upgrades
    """

    def __init__(self):
        self.module_args = dict(name=dict(type='str', required=True), resource_group=dict(type='str', required=True))
        self.results = dict(changed=False, azure_aks_upgrades=[])
        self.name = None
        self.resource_group = None
        super(AzureRMAKSUpgrade, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.results['azure_aks_upgrades'] = self.get_upgrades(self.name, self.resource_group)
        return self.results

    def get_upgrades(self, name, resource_group):
        """
        Get supported upgrade version for AKS
        :param: name: str with name of AKS cluster instance
        :param: resource_group: str with resource group containing AKS instance
        :return: dict with available versions for pool profiles and control plane
        """
        cluster = None
        upgrade_profiles = None
        self.log('Get properties for {0}'.format(self.name))
        try:
            cluster = self.managedcluster_client.managed_clusters.get(resource_group_name=resource_group, resource_name=name)
        except ResourceNotFoundError as err:
            self.fail('Error when getting AKS cluster information for {0} : {1}'.format(self.name, err.message or str(err)))
        self.log('Get available upgrade versions for {0}'.format(self.name))
        try:
            upgrade_profiles = self.managedcluster_client.managed_clusters.get_upgrade_profile(resource_group_name=resource_group, resource_name=name)
        except ResourceNotFoundError as err:
            self.fail('Error when getting upgrade versions for {0} : {1}'.format(self.name, err.message or str(err)))
        return dict(agent_pool_profiles=[self.parse_profile(profile) if profile.upgrades else self.default_profile(cluster) for profile in upgrade_profiles.agent_pool_profiles] if upgrade_profiles.agent_pool_profiles else None, control_plane_profile=self.parse_profile(upgrade_profiles.control_plane_profile) if upgrade_profiles.control_plane_profile.upgrades else self.default_profile(cluster))

    def default_profile(self, cluster):
        """
        Used when upgrade profile returned by Azure in None
        (i.e. when the cluster runs latest version)
        :param: cluster: ManagedCluster with AKS instance information
        :return: dict containing upgrade profile with current cluster version
        """
        return dict(upgrades=None, kubernetes_version=cluster.kubernetes_version, name=None, os_type=None)

    def parse_profile(self, profile):
        """
        Transform cluster profile object to dict
        :param: profile: ManagedClusterUpgradeProfile with AKS upgrade profile info
        :return: dict with upgrade profiles
        """
        return dict(upgrades=[dict(is_preview=upgrade.is_preview, kubernetes_version=upgrade.kubernetes_version) for upgrade in profile.upgrades], kubernetes_version=profile.kubernetes_version, name=profile.name, os_type=profile.os_type)