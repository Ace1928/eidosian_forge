from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMManagedCluster(AzureRMModuleBase):
    """Configuration class for an Azure RM container service (AKS) resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), dns_prefix=dict(type='str'), kubernetes_version=dict(type='str'), linux_profile=dict(type='dict', options=linux_profile_spec), agent_pool_profiles=dict(type='list', elements='dict', options=agent_pool_profile_spec), service_principal=dict(type='dict', options=service_principal_spec), enable_rbac=dict(type='bool', default=False), network_profile=dict(type='dict', options=network_profile_spec), aad_profile=dict(type='dict', options=aad_profile_spec), addon=dict(type='dict', options=create_addon_profiles_spec()), api_server_access_profile=dict(type='dict', options=api_server_access_profile_spec), node_resource_group=dict(type='str'))
        self.resource_group = None
        self.name = None
        self.location = None
        self.dns_prefix = None
        self.kubernetes_version = None
        self.tags = None
        self.state = None
        self.linux_profile = None
        self.agent_pool_profiles = None
        self.service_principal = None
        self.enable_rbac = False
        self.network_profile = None
        self.aad_profile = None
        self.api_server_access_profile = None
        self.addon = None
        self.node_resource_group = None
        required_if = [('state', 'present', ['dns_prefix', 'agent_pool_profiles'])]
        self.results = dict(changed=False)
        super(AzureRMManagedCluster, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        resource_group = None
        to_be_updated = False
        update_tags = False
        update_agentpool = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        response = self.get_aks()
        if self.state == 'present':
            available_versions = self.get_all_versions()
            if not response:
                to_be_updated = True
                if self.kubernetes_version not in available_versions.keys():
                    self.fail('Unsupported kubernetes version. Expected one of {0} but got {1}'.format(available_versions.keys(), self.kubernetes_version))
            else:
                self.results = response
                self.results['changed'] = False
                self.log('Results : {0}'.format(response))
                update_tags, response['tags'] = self.update_tags(response['tags'])
                if response['provisioning_state'] == 'Succeeded':

                    def is_property_changed(profile, property, ignore_case=False):
                        base = response[profile].get(property)
                        new = getattr(self, profile).get(property)
                        if ignore_case:
                            return base.lower() != new.lower()
                        else:
                            return base != new
                    if self.linux_profile and is_property_changed('linux_profile', 'ssh_key'):
                        self.log('Linux Profile Diff SSH, Was {0} / Now {1}'.format(response['linux_profile']['ssh_key'], self.linux_profile.get('ssh_key')))
                        to_be_updated = True
                    if self.linux_profile and is_property_changed('linux_profile', 'admin_username'):
                        self.log('Linux Profile Diff User, Was {0} / Now {1}'.format(response['linux_profile']['admin_username'], self.linux_profile.get('admin_username')))
                        to_be_updated = True
                    if len(response['agent_pool_profiles']) != len(self.agent_pool_profiles):
                        self.log('Agent Pool count is diff, need to update')
                        update_agentpool = True
                    if response['kubernetes_version'] != self.kubernetes_version:
                        upgrade_versions = available_versions.get(response['kubernetes_version']) or available_versions.keys()
                        if upgrade_versions and self.kubernetes_version not in upgrade_versions:
                            self.fail('Cannot upgrade kubernetes version to {0}, supported value are {1}'.format(self.kubernetes_version, upgrade_versions))
                        to_be_updated = True
                    if response['enable_rbac'] != self.enable_rbac:
                        to_be_updated = True
                    if response['api_server_access_profile'] != self.api_server_access_profile and self.api_server_access_profile is not None:
                        if self.api_server_access_profile.get('enable_private_cluster') != response['api_server_access_profile'].get('enable_private_cluster'):
                            self.log('Api Server Access Diff - Origin {0} / Update {1}'.format(str(self.api_server_access_profile), str(response['api_server_access_profile'])))
                            self.fail('The enable_private_cluster of the api server access profile cannot be updated')
                        elif self.api_server_access_profile.get('authorized_ip_ranges') is not None and len(self.api_server_access_profile.get('authorized_ip_ranges')) != len(response['api_server_access_profile'].get('authorized_ip_ranges', [])):
                            self.log('Api Server Access Diff - Origin {0} / Update {1}'.format(str(self.api_server_access_profile), str(response['api_server_access_profile'])))
                            to_be_updated = True
                    if self.network_profile:
                        for key in self.network_profile.keys():
                            original = response['network_profile'].get(key) or ''
                            if self.network_profile[key] and self.network_profile[key].lower() != original.lower():
                                to_be_updated = True

                    def compare_addon(origin, patch, config):
                        if not patch:
                            return True
                        if not origin:
                            return False
                        if origin['enabled'] != patch['enabled']:
                            return False
                        config = config or dict()
                        for key in config.keys():
                            if origin.get(config[key]) != patch.get(key):
                                return False
                        return True
                    if self.addon:
                        for key in ADDONS.keys():
                            addon_name = ADDONS[key]['name']
                            if not compare_addon(response['addon'].get(addon_name), self.addon.get(key), ADDONS[key].get('config')):
                                to_be_updated = True
                    for profile_result in response['agent_pool_profiles']:
                        matched = False
                        for profile_self in self.agent_pool_profiles:
                            if profile_result['name'] == profile_self['name']:
                                matched = True
                                os_disk_size_gb = profile_self.get('os_disk_size_gb') or profile_result['os_disk_size_gb']
                                vnet_subnet_id = profile_self.get('vnet_subnet_id', profile_result['vnet_subnet_id'])
                                count = profile_self['count']
                                orchestrator_version = profile_self['orchestrator_version']
                                vm_size = profile_self['vm_size']
                                availability_zones = profile_self['availability_zones']
                                enable_auto_scaling = profile_self['enable_auto_scaling']
                                mode = profile_self['mode']
                                max_count = profile_self['max_count']
                                node_labels = profile_self['node_labels']
                                min_count = profile_self['min_count']
                                max_pods = profile_self['max_pods']
                                if max_pods is not None and profile_result['max_pods'] != max_pods:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    self.fail('The max_pods of the agent pool cannot be updated')
                                elif vnet_subnet_id is not None and profile_result['vnet_subnet_id'] != vnet_subnet_id:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    self.fail('The vnet_subnet_id of the agent pool cannot be updated')
                                elif availability_zones is not None and ' '.join(map(str, profile_result['availability_zones'])) != ' '.join(map(str, availability_zones)):
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    self.fail('The availability_zones of the agent pool cannot be updated')
                                if count is not None and profile_result['count'] != count:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif vm_size is not None and profile_result['vm_size'] != vm_size:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif os_disk_size_gb is not None and profile_result['os_disk_size_gb'] != os_disk_size_gb:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif enable_auto_scaling is not None and profile_result['enable_auto_scaling'] != enable_auto_scaling:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif max_count is not None and profile_result['max_count'] != max_count:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif min_count is not None and profile_result['min_count'] != min_count:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif mode is not None and profile_result['mode'] != mode:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                                elif node_labels is not None and profile_result['node_labels'] != node_labels:
                                    self.log('Agent Profile Diff - Origin {0} / Update {1}'.format(str(profile_result), str(profile_self)))
                                    to_be_updated = True
                        if not matched:
                            self.log('Agent Pool not found')
                            to_be_updated = True
            if update_agentpool:
                self.log('Need to update agentpool')
                if not self.check_mode:
                    response_profile_name_list = [response_profile['name'] for response_profile in response['agent_pool_profiles']]
                    self_profile_name_list = [self_profile['name'] for self_profile in self.agent_pool_profiles]
                    to_update = list(set(self_profile_name_list) - set(response_profile_name_list))
                    to_delete = list(set(response_profile_name_list) - set(self_profile_name_list))
                    if len(to_delete) > 0:
                        self.delete_agentpool(to_delete)
                        for profile in self.results['agent_pool_profiles']:
                            if profile['name'] in to_delete:
                                self.results['agent_pool_profiles'].remove(profile)
                    if len(to_update) > 0:
                        self.results['agent_pool_profiles'].extend(self.create_update_agentpool(to_update))
                    self.log('Creation / Update done')
                self.results['changed'] = True
            if to_be_updated:
                self.log('Need to Create / Update the AKS instance')
                if not self.check_mode:
                    self.results = self.create_update_aks()
                    self.log('Creation / Update done')
                self.results['changed'] = True
            elif update_tags:
                self.log('Need to Update the AKS tags')
                if not self.check_mode:
                    self.results['tags'] = self.update_aks_tags()
                self.results['changed'] = True
            return self.results
        elif self.state == 'absent' and response:
            self.log('Need to Delete the AKS instance')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_aks()
            self.log('AKS instance deleted')
        return self.results

    def create_update_aks(self):
        """
        Creates or updates a managed Azure container service (AKS) with the specified configuration of agents.

        :return: deserialized AKS instance state dictionary
        """
        self.log('Creating / Updating the AKS instance {0}'.format(self.name))
        agentpools = []
        if self.agent_pool_profiles:
            agentpools = [self.create_agent_pool_profile_instance(profile) for profile in self.agent_pool_profiles]
        if self.service_principal:
            service_principal_profile = self.create_service_principal_profile_instance(self.service_principal)
            identity = None
        else:
            service_principal_profile = None
            identity = self.managedcluster_models.ManagedClusterIdentity(type='SystemAssigned')
        if self.linux_profile:
            linux_profile = self.create_linux_profile_instance(self.linux_profile)
        else:
            linux_profile = None
        parameters = self.managedcluster_models.ManagedCluster(location=self.location, dns_prefix=self.dns_prefix, kubernetes_version=self.kubernetes_version, tags=self.tags, service_principal_profile=service_principal_profile, agent_pool_profiles=agentpools, linux_profile=linux_profile, identity=identity, enable_rbac=self.enable_rbac, network_profile=self.create_network_profile_instance(self.network_profile), aad_profile=self.create_aad_profile_instance(self.aad_profile), api_server_access_profile=self.create_api_server_access_profile_instance(self.api_server_access_profile), addon_profiles=self.create_addon_profile_instance(self.addon), node_resource_group=self.node_resource_group)
        try:
            poller = self.managedcluster_client.managed_clusters.begin_create_or_update(self.resource_group, self.name, parameters)
            response = self.get_poller_result(poller)
            response.kube_config = self.get_aks_kubeconfig()
            return create_aks_dict(response)
        except Exception as exc:
            self.log('Error attempting to create the AKS instance.')
            self.fail('Error creating the AKS instance: {0}'.format(exc.message))

    def update_aks_tags(self):
        try:
            poller = self.managedcluster_client.managed_clusters.begin_update_tags(self.resource_group, self.name, self.tags)
            response = self.get_poller_result(poller)
            return response.tags
        except Exception as exc:
            self.fail('Error attempting to update AKS tags: {0}'.format(exc.message))

    def create_update_agentpool(self, to_update_name_list):
        response_all = []
        for profile in self.agent_pool_profiles:
            if profile['name'] in to_update_name_list:
                self.log('Creating / Updating the AKS agentpool {0}'.format(profile['name']))
                parameters = self.managedcluster_models.AgentPool(count=profile['count'], vm_size=profile['vm_size'], os_disk_size_gb=profile['os_disk_size_gb'], max_count=profile['max_count'], node_labels=profile['node_labels'], min_count=profile['min_count'], orchestrator_version=profile['orchestrator_version'], max_pods=profile['max_pods'], enable_auto_scaling=profile['enable_auto_scaling'], agent_pool_type=profile['type'], mode=profile['mode'])
                try:
                    poller = self.managedcluster_client.agent_pools.begin_create_or_update(self.resource_group, self.name, profile['name'], parameters)
                    response = self.get_poller_result(poller)
                    response_all.append(response)
                except Exception as exc:
                    self.fail('Error attempting to update AKS agentpool: {0}'.format(exc.message))
        return create_agent_pool_profiles_dict(response_all)

    def delete_agentpool(self, to_delete_name_list):
        for name in to_delete_name_list:
            self.log('Deleting the AKS agentpool {0}'.format(name))
            try:
                poller = self.managedcluster_client.agent_pools.begin_delete(self.resource_group, self.name, name)
                self.get_poller_result(poller)
            except Exception as exc:
                self.fail('Error attempting to update AKS agentpool: {0}'.format(exc.message))

    def delete_aks(self):
        """
        Deletes the specified managed container service (AKS) in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the AKS instance {0}'.format(self.name))
        try:
            poller = self.managedcluster_client.managed_clusters.begin_delete(self.resource_group, self.name)
            self.get_poller_result(poller)
            return True
        except Exception as e:
            self.log('Error attempting to delete the AKS instance.')
            self.fail('Error deleting the AKS instance: {0}'.format(e.message))
            return False

    def get_aks(self):
        """
        Gets the properties of the specified container service.

        :return: deserialized AKS instance state dictionary
        """
        self.log('Checking if the AKS instance {0} is present'.format(self.name))
        try:
            response = self.managedcluster_client.managed_clusters.get(self.resource_group, self.name)
            self.log('Response : {0}'.format(response))
            self.log('AKS instance : {0} found'.format(response.name))
            response.kube_config = self.get_aks_kubeconfig()
            return create_aks_dict(response)
        except ResourceNotFoundError:
            self.log('Did not find the AKS instance.')
            return False

    def get_all_versions(self):
        try:
            result = dict()
            response = self.containerservice_client.container_services.list_orchestrators(self.location, resource_type='managedClusters')
            orchestrators = response.orchestrators
            for item in orchestrators:
                result[item.orchestrator_version] = [x.orchestrator_version for x in item.upgrades] if item.upgrades else []
            return result
        except Exception as exc:
            self.fail('Error when getting AKS supported kubernetes version list for location {0} - {1}'.format(self.location, exc.message or str(exc)))

    def get_aks_kubeconfig(self):
        """
        Gets kubeconfig for the specified AKS instance.

        :return: AKS instance kubeconfig
        """
        access_profile = self.managedcluster_client.managed_clusters.get_access_profile(resource_group_name=self.resource_group, resource_name=self.name, role_name='clusterUser')
        return access_profile.kube_config.decode('utf-8')

    def create_agent_pool_profile_instance(self, agentpoolprofile):
        """
        Helper method to serialize a dict to a ManagedClusterAgentPoolProfile
        :param: agentpoolprofile: dict with the parameters to setup the ManagedClusterAgentPoolProfile
        :return: ManagedClusterAgentPoolProfile
        """
        return self.managedcluster_models.ManagedClusterAgentPoolProfile(**agentpoolprofile)

    def create_service_principal_profile_instance(self, spnprofile):
        """
        Helper method to serialize a dict to a ManagedClusterServicePrincipalProfile
        :param: spnprofile: dict with the parameters to setup the ManagedClusterServicePrincipalProfile
        :return: ManagedClusterServicePrincipalProfile
        """
        return self.managedcluster_models.ManagedClusterServicePrincipalProfile(client_id=spnprofile['client_id'], secret=spnprofile['client_secret'])

    def create_linux_profile_instance(self, linuxprofile):
        """
        Helper method to serialize a dict to a ContainerServiceLinuxProfile
        :param: linuxprofile: dict with the parameters to setup the ContainerServiceLinuxProfile
        :return: ContainerServiceLinuxProfile
        """
        return self.managedcluster_models.ContainerServiceLinuxProfile(admin_username=linuxprofile['admin_username'], ssh=self.managedcluster_models.ContainerServiceSshConfiguration(public_keys=[self.managedcluster_models.ContainerServiceSshPublicKey(key_data=str(linuxprofile['ssh_key']))]))

    def create_network_profile_instance(self, network):
        return self.managedcluster_models.ContainerServiceNetworkProfile(**network) if network else None

    def create_api_server_access_profile_instance(self, server_access):
        return self.managedcluster_models.ManagedClusterAPIServerAccessProfile(**server_access) if server_access else None

    def create_aad_profile_instance(self, aad):
        return self.managedcluster_models.ManagedClusterAADProfile(**aad) if aad else None

    def create_addon_profile_instance(self, addon):
        result = dict()
        addon = addon or {}
        for key in addon.keys():
            if not ADDONS.get(key):
                self.fail('Unsupported addon {0}'.format(key))
            if addon.get(key):
                name = ADDONS[key]['name']
                config_spec = ADDONS[key].get('config') or dict()
                config = addon[key]
                for v in config_spec.keys():
                    config[config_spec[v]] = config[v]
                result[name] = self.managedcluster_models.ManagedClusterAddonProfile(config=config, enabled=config['enabled'])
        return result