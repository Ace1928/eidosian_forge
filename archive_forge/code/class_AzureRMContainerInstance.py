from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMContainerInstance(AzureRMModuleBase):
    """Configuration class for an Azure RM container instance resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), os_type=dict(type='str', default='linux', choices=['linux', 'windows']), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), ip_address=dict(type='str', default='none', choices=['public', 'none', 'private']), dns_name_label=dict(type='str'), ports=dict(type='list', elements='int', default=[]), registry_login_server=dict(type='str', default=None), registry_username=dict(type='str', default=None), registry_password=dict(type='str', default=None, no_log=True), containers=dict(type='list', elements='dict', options=container_spec), restart_policy=dict(type='str', choices=['always', 'on_failure', 'never']), force_update=dict(type='bool', default=False), volumes=dict(type='list', elements='dict', options=volumes_spec), subnet_ids=dict(type='list', elements='str'))
        self.resource_group = None
        self.name = None
        self.location = None
        self.state = None
        self.ip_address = None
        self.dns_name_label = None
        self.containers = None
        self.restart_policy = None
        self.subnet_ids = None
        self.tags = None
        self.results = dict(changed=False, state=dict())
        self.cgmodels = None
        required_if = [('state', 'present', ['containers']), ('ip_address', 'private', ['subnet_ids'])]
        super(AzureRMContainerInstance, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        resource_group = None
        response = None
        results = dict()
        self.cgmodels = self.containerinstance_client.container_groups.models
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        response = self.get_containerinstance()
        if not response:
            self.log("Container Group doesn't exist")
            if self.state == 'absent':
                self.log('Nothing to delete')
            else:
                self.force_update = True
        else:
            self.log('Container instance already exists')
            if self.state == 'absent':
                if not self.check_mode:
                    self.delete_containerinstance()
                self.results['changed'] = True
                self.log('Container instance deleted')
            elif self.state == 'present':
                self.log('Need to check if container group has to be deleted or may be updated')
                update_tags, newtags = self.update_tags(response.get('tags', dict()))
                if self.force_update:
                    self.log('Deleting container instance before update')
                    if not self.check_mode:
                        self.delete_containerinstance()
                elif update_tags:
                    if not self.check_mode:
                        self.tags = newtags
                        self.results['changed'] = True
                        response = self.update_containerinstance()
        if self.state == 'present':
            self.log('Need to Create / Update the container instance')
            if self.force_update:
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                response = self.create_update_containerinstance()
            self.results['id'] = response['id']
            self.results['provisioning_state'] = response['provisioning_state']
            self.results['ip_address'] = response['ip_address']['ip'] if 'ip_address' in response else ''
            self.log('Creation / Update done')
        return self.results

    def update_containerinstance(self):
        """
        Updates a container service with the specified configuration of orchestrator, masters, and agents.

        :return: deserialized container instance state dictionary
        """
        try:
            response = self.containerinstance_client.container_groups.update(resource_group_name=self.resource_group, container_group_name=self.name, resource=dict(tags=self.tags))
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error when Updating ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
        return response.as_dict()

    def create_update_containerinstance(self):
        """
        Creates or updates a container service with the specified configuration of orchestrator, masters, and agents.

        :return: deserialized container instance state dictionary
        """
        self.log('Creating / Updating the container instance {0}'.format(self.name))
        registry_credentials = None
        if self.registry_login_server is not None:
            registry_credentials = [self.cgmodels.ImageRegistryCredential(server=self.registry_login_server, username=self.registry_username, password=self.registry_password)]
        ip_address = None
        containers = []
        all_ports = set([])
        for container_def in self.containers:
            name = container_def.get('name')
            image = container_def.get('image')
            memory = container_def.get('memory')
            cpu = container_def.get('cpu')
            commands = container_def.get('commands')
            ports = []
            variables = []
            volume_mounts = []
            port_list = container_def.get('ports')
            if port_list:
                for port in port_list:
                    all_ports.add(port)
                    ports.append(self.cgmodels.ContainerPort(port=port))
            variable_list = container_def.get('environment_variables')
            if variable_list:
                for variable in variable_list:
                    variables.append(self.cgmodels.EnvironmentVariable(name=variable.get('name'), value=variable.get('value') if not variable.get('is_secure') else None, secure_value=variable.get('value') if variable.get('is_secure') else None))
            volume_mounts_list = container_def.get('volume_mounts')
            if volume_mounts_list:
                for volume_mount in volume_mounts_list:
                    volume_mounts.append(self.cgmodels.VolumeMount(name=volume_mount.get('name'), mount_path=volume_mount.get('mount_path'), read_only=volume_mount.get('read_only')))
            containers.append(self.cgmodels.Container(name=name, image=image, resources=self.cgmodels.ResourceRequirements(requests=self.cgmodels.ResourceRequests(memory_in_gb=memory, cpu=cpu)), ports=ports, command=commands, environment_variables=variables, volume_mounts=volume_mounts))
        if self.ip_address is not None:
            if len(all_ports) > 0:
                ports = []
                for port in all_ports:
                    ports.append(self.cgmodels.Port(port=port, protocol='TCP'))
                ip_address = self.cgmodels.IpAddress(ports=ports, dns_name_label=self.dns_name_label, type=self.ip_address)
        subnet_ids = None
        if self.subnet_ids is not None:
            subnet_ids = [self.cgmodels.ContainerGroupSubnetId(id=item) for item in self.subnet_ids]
        parameters = self.cgmodels.ContainerGroup(location=self.location, containers=containers, image_registry_credentials=registry_credentials, restart_policy=_snake_to_camel(self.restart_policy, True) if self.restart_policy else None, ip_address=ip_address, os_type=self.os_type, subnet_ids=subnet_ids, volumes=self.volumes, tags=self.tags)
        try:
            response = self.containerinstance_client.container_groups.begin_create_or_update(resource_group_name=self.resource_group, container_group_name=self.name, container_group=parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error when creating ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
        return response.as_dict()

    def delete_containerinstance(self):
        """
        Deletes the specified container group instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the container instance {0}'.format(self.name))
        try:
            response = self.containerinstance_client.container_groups.begin_delete(resource_group_name=self.resource_group, container_group_name=self.name)
            return True
        except Exception as exc:
            self.fail('Error when deleting ACI {0}: {1}'.format(self.name, exc.message or str(exc)))
            return False

    def get_containerinstance(self):
        """
        Gets the properties of the specified container service.

        :return: deserialized container instance state dictionary
        """
        self.log('Checking if the container instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.containerinstance_client.container_groups.get(resource_group_name=self.resource_group, container_group_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Container instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the container instance.')
        if found is True:
            return response.as_dict()
        return False