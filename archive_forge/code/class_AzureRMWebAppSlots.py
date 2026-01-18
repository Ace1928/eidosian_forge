from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMWebAppSlots(AzureRMModuleBase):
    """Configuration class for an Azure RM Web App slot resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), webapp_name=dict(type='str', required=True), location=dict(type='str'), configuration_source=dict(type='str'), auto_swap_slot_name=dict(type='raw'), swap=dict(type='dict', options=swap_spec), frameworks=dict(type='list', elements='dict', options=framework_spec), container_settings=dict(type='dict', options=container_settings_spec), deployment_source=dict(type='dict', options=deployment_source_spec), startup_file=dict(type='str'), app_settings=dict(type='dict'), purge_app_settings=dict(type='bool', default=False), app_state=dict(type='str', choices=['started', 'stopped', 'restarted'], default='started'), state=dict(type='str', default='present', choices=['present', 'absent']))
        mutually_exclusive = [['container_settings', 'frameworks']]
        self.resource_group = None
        self.name = None
        self.webapp_name = None
        self.location = None
        self.auto_swap_slot_name = None
        self.swap = None
        self.tags = None
        self.startup_file = None
        self.configuration_source = None
        self.clone = False
        self.site_config = dict()
        self.app_settings = dict()
        self.app_settings_strDic = None
        self.deployment_source = dict()
        self.site = None
        self.container_settings = None
        self.purge_app_settings = False
        self.app_state = 'started'
        self.results = dict(changed=False, id=None)
        self.state = None
        self.to_do = Actions.NoAction
        self.frameworks = None
        self.site_config_updatable_frameworks = ['net_framework_version', 'java_version', 'php_version', 'python_version', 'linux_fx_version']
        self.supported_linux_frameworks = ['ruby', 'php', 'dotnetcore', 'node', 'java']
        self.supported_windows_frameworks = ['net_framework', 'php', 'python', 'node', 'java']
        super(AzureRMWebAppSlots, self).__init__(derived_arg_spec=self.module_arg_spec, mutually_exclusive=mutually_exclusive, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'scm_type':
                    self.site_config[key] = kwargs[key]
        old_response = None
        response = None
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        webapp_response = self.get_webapp()
        if not webapp_response:
            self.fail('Web app {0} does not exist in resource group {1}.'.format(self.webapp_name, self.resource_group))
        old_response = self.get_slot()
        is_linux = True if webapp_response['reserved'] else False
        if self.state == 'present':
            if self.frameworks:
                if len(self.frameworks) > 1 and any((f['name'] == 'java' for f in self.frameworks)):
                    self.fail('Java is mutually exclusive with other frameworks.')
                if is_linux:
                    if len(self.frameworks) != 1:
                        self.fail('Can specify one framework only for Linux web app.')
                    if self.frameworks[0]['name'] not in self.supported_linux_frameworks:
                        self.fail('Unsupported framework {0} for Linux web app.'.format(self.frameworks[0]['name']))
                    self.site_config['linux_fx_version'] = (self.frameworks[0]['name'] + '|' + self.frameworks[0]['version']).upper()
                    if self.frameworks[0]['name'] == 'java':
                        if self.frameworks[0]['version'] != '8':
                            self.fail('Linux web app only supports java 8.')
                        if self.frameworks[0].get('settings', {}) and self.frameworks[0]['settings'].get('java_container', None) and (self.frameworks[0]['settings']['java_container'].lower() != 'tomcat'):
                            self.fail('Linux web app only supports tomcat container.')
                        if self.frameworks[0].get('settings', {}) and self.frameworks[0]['settings'].get('java_container', None) and (self.frameworks[0]['settings']['java_container'].lower() == 'tomcat'):
                            self.site_config['linux_fx_version'] = 'TOMCAT|' + self.frameworks[0]['settings']['java_container_version'] + '-jre8'
                        else:
                            self.site_config['linux_fx_version'] = 'JAVA|8-jre8'
                else:
                    for fx in self.frameworks:
                        if fx.get('name') not in self.supported_windows_frameworks:
                            self.fail('Unsupported framework {0} for Windows web app.'.format(fx.get('name')))
                        else:
                            self.site_config[fx.get('name') + '_version'] = fx.get('version')
                        if 'settings' in fx and fx['settings'] is not None:
                            for key, value in fx['settings'].items():
                                self.site_config[key] = value
            if not self.app_settings:
                self.app_settings = dict()
            if self.container_settings:
                linux_fx_version = 'DOCKER|'
                if self.container_settings.get('registry_server_url'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_URL'] = 'https://' + self.container_settings['registry_server_url']
                    linux_fx_version += self.container_settings['registry_server_url'] + '/'
                linux_fx_version += self.container_settings['name']
                self.site_config['linux_fx_version'] = linux_fx_version
                if self.container_settings.get('registry_server_user'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_USERNAME'] = self.container_settings['registry_server_user']
                if self.container_settings.get('registry_server_password'):
                    self.app_settings['DOCKER_REGISTRY_SERVER_PASSWORD'] = self.container_settings['registry_server_password']
            if self.auto_swap_slot_name and isinstance(self.auto_swap_slot_name, str):
                self.site_config['auto_swap_slot_name'] = self.auto_swap_slot_name
            if self.auto_swap_slot_name is False:
                self.site_config['auto_swap_slot_name'] = None
            self.site = Site(location=self.location, site_config=self.site_config)
            if not old_response:
                self.log("Web App slot doesn't exist")
                to_be_updated = True
                self.to_do = Actions.CreateOrUpdate
                self.site.tags = self.tags
                if self.startup_file:
                    self.site_config['app_command_line'] = self.startup_file
                if self.app_settings:
                    app_settings = []
                    for key in self.app_settings.keys():
                        app_settings.append(NameValuePair(name=key, value=self.app_settings[key]))
                    self.site_config['app_settings'] = app_settings
                if self.configuration_source:
                    self.clone = True
            else:
                self.log('Web App slot already exists')
                self.log('Result: {0}'.format(old_response))
                update_tags, self.site.tags = self.update_tags(old_response.get('tags', None))
                if update_tags:
                    to_be_updated = True
                old_config = self.get_configuration_slot(self.name)
                if self.is_site_config_changed(old_config):
                    to_be_updated = True
                    self.to_do = Actions.CreateOrUpdate
                self.app_settings_strDic = self.list_app_settings_slot(self.name)
                if self.purge_app_settings:
                    to_be_updated = True
                    self.to_do = Actions.UpdateAppSettings
                    self.app_settings_strDic = dict()
                if self.purge_app_settings or self.is_app_settings_changed():
                    to_be_updated = True
                    self.to_do = Actions.UpdateAppSettings
                    if self.app_settings:
                        for key in self.app_settings.keys():
                            self.app_settings_strDic[key] = self.app_settings[key]
        elif self.state == 'absent':
            if old_response:
                self.log('Delete Web App slot')
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.delete_slot()
                self.log('Web App slot deleted')
            else:
                self.log('Web app slot {0} not exists.'.format(self.name))
        if to_be_updated:
            self.log('Need to Create/Update web app')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            if self.to_do == Actions.CreateOrUpdate:
                response = self.create_update_slot()
                self.results['id'] = response['id']
                if self.clone:
                    self.clone_slot()
            if self.to_do == Actions.UpdateAppSettings:
                self.update_app_settings_slot()
        slot = None
        if response:
            slot = response
        if old_response:
            slot = old_response
        if slot:
            if slot['state'] != 'Stopped' and self.app_state == 'stopped' or (slot['state'] != 'Running' and self.app_state == 'started') or self.app_state == 'restarted':
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.set_state_slot(self.app_state)
            if self.swap:
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                self.swap_slot()
        return self.results

    def is_site_config_changed(self, existing_config):
        for fx_version in self.site_config_updatable_frameworks:
            if self.site_config.get(fx_version):
                if not getattr(existing_config, fx_version) or getattr(existing_config, fx_version).upper() != self.site_config.get(fx_version).upper():
                    return True
        if self.auto_swap_slot_name is False and existing_config.auto_swap_slot_name is not None:
            return True
        elif self.auto_swap_slot_name and self.auto_swap_slot_name != getattr(existing_config, 'auto_swap_slot_name', None):
            return True
        return False

    def is_app_settings_changed(self):
        if self.app_settings:
            if len(self.app_settings_strDic) != len(self.app_settings):
                return True
            if self.app_settings_strDic != self.app_settings:
                return True
        return False

    def is_deployment_source_changed(self, existing_webapp):
        if self.deployment_source:
            if self.deployment_source.get('url') and self.deployment_source['url'] != existing_webapp.get('site_source_control')['url']:
                return True
            if self.deployment_source.get('branch') and self.deployment_source['branch'] != existing_webapp.get('site_source_control')['branch']:
                return True
        return False

    def create_update_slot(self):
        """
        Creates or updates Web App slot with the specified configuration.

        :return: deserialized Web App instance state dictionary
        """
        self.log('Creating / Updating the Web App slot {0}'.format(self.name))
        try:
            response = self.web_client.web_apps.begin_create_or_update_slot(resource_group_name=self.resource_group, slot=self.name, name=self.webapp_name, site_envelope=self.site)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Web App slot instance.')
            self.fail('Error creating the Web App slot: {0}'.format(str(exc)))
        return slot_to_dict(response)

    def delete_slot(self):
        """
        Deletes specified Web App slot in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Web App slot {0}'.format(self.name))
        try:
            self.web_client.web_apps.delete_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Web App slot.')
            self.fail('Error deleting the Web App slots: {0}'.format(str(e)))
        return True

    def get_webapp(self):
        """
        Gets the properties of the specified Web App.

        :return: deserialized Web App instance state dictionary
        """
        self.log('Checking if the Web App instance {0} is present'.format(self.webapp_name))
        response = None
        try:
            response = self.web_client.web_apps.get(resource_group_name=self.resource_group, name=self.webapp_name)
            if response is not None:
                self.log('Response : {0}'.format(response))
                self.log('Web App instance : {0} found'.format(response.name))
                return webapp_to_dict(response)
        except ResourceNotFoundError:
            pass
        self.log("Didn't find web app {0} in resource group {1}".format(self.webapp_name, self.resource_group))
        return False

    def get_slot(self):
        """
        Gets the properties of the specified Web App slot.

        :return: deserialized Web App slot state dictionary
        """
        self.log('Checking if the Web App slot {0} is present'.format(self.name))
        response = None
        try:
            response = self.web_client.web_apps.get_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
            if response is not None:
                self.log('Response : {0}'.format(response))
                self.log('Web App slot: {0} found'.format(response.name))
                return slot_to_dict(response)
        except ResourceNotFoundError:
            pass
        self.log('Does not find web app slot {0} in resource group {1}'.format(self.name, self.resource_group))
        return False

    def list_app_settings(self):
        """
        List webapp application settings
        :return: deserialized list response
        """
        self.log('List webapp application setting')
        try:
            response = self.web_client.web_apps.list_application_settings(resource_group_name=self.resource_group, name=self.webapp_name)
            self.log('Response : {0}'.format(response))
            return response.properties
        except Exception as ex:
            self.fail('Failed to list application settings for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def list_app_settings_slot(self, slot_name):
        """
        List application settings
        :return: deserialized list response
        """
        self.log('List application setting')
        try:
            response = self.web_client.web_apps.list_application_settings_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name)
            self.log('Response : {0}'.format(response))
            return response.properties
        except Exception as ex:
            self.fail('Failed to list application settings for web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def update_app_settings_slot(self, slot_name=None, app_settings=None):
        """
        Update application settings
        :return: deserialized updating response
        """
        self.log('Update application setting')
        if slot_name is None:
            slot_name = self.name
        if app_settings is None:
            app_settings = self.app_settings_strDic
        try:
            settings = StringDictionary(properties=self.app_settings)
            response = self.web_client.web_apps.update_application_settings_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name, app_settings=settings)
            self.log('Response : {0}'.format(response))
            return response.as_dict()
        except Exception as ex:
            self.fail('Failed to update application settings for web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return response

    def create_or_update_source_control_slot(self):
        """
        Update site source control
        :return: deserialized updating response
        """
        self.log('Update site source control')
        if self.deployment_source is None:
            return False
        self.deployment_source['is_manual_integration'] = False
        self.deployment_source['is_mercurial'] = False
        try:
            site_source_control = SiteSourceControl(repo_url=self.deployment_source.get('url'), branch=self.deployment_source.get('branch'))
            response = self.web_client.web_client.begin_create_or_update_source_control_slot(resource_group_name=self.resource_group, name=self.webapp_name, site_source_control=site_source_control, slot=self.name)
            self.log('Response : {0}'.format(response))
            return response.as_dict()
        except Exception as ex:
            self.fail('Failed to update site source control for web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def get_configuration(self):
        """
        Get web app configuration
        :return: deserialized web app configuration response
        """
        self.log('Get web app configuration')
        try:
            response = self.web_client.web_apps.get_configuration(resource_group_name=self.resource_group, name=self.webapp_name)
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            self.fail('Failed to get configuration for web app {0} in resource group {1}: {2}'.format(self.webapp_name, self.resource_group, str(ex)))

    def get_configuration_slot(self, slot_name):
        """
        Get slot configuration
        :return: deserialized slot configuration response
        """
        self.log('Get web app slot configuration')
        try:
            response = self.web_client.web_apps.get_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name)
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            self.fail('Failed to get configuration for web app slot {0} in resource group {1}: {2}'.format(slot_name, self.resource_group, str(ex)))

    def update_configuration_slot(self, slot_name=None, site_config=None):
        """
        Update slot configuration
        :return: deserialized slot configuration response
        """
        self.log('Update web app slot configuration')
        if slot_name is None:
            slot_name = self.name
        if site_config is None:
            site_config = self.site_config
        try:
            response = self.web_client.web_apps.update_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name, site_config=site_config)
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            self.fail('Failed to update configuration for web app slot {0} in resource group {1}: {2}'.format(slot_name, self.resource_group, str(ex)))

    def set_state_slot(self, appstate):
        """
        Start/stop/restart web app slot
        :return: deserialized updating response
        """
        try:
            if appstate == 'started':
                response = self.web_client.web_apps.start_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
            elif appstate == 'stopped':
                response = self.web_client.web_apps.stop_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
            elif appstate == 'restarted':
                response = self.web_client.web_apps.restart_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
            else:
                self.fail('Invalid web app slot state {0}'.format(appstate))
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            request_id = ex.request_id if ex.request_id else ''
            self.fail('Failed to {0} web app slot {1} in resource group {2}, request_id {3} - {4}'.format(appstate, self.name, self.resource_group, request_id, str(ex)))

    def swap_slot(self):
        """
        Swap slot
        :return: deserialized response
        """
        self.log('Swap slot')
        try:
            if self.swap['action'] == 'swap':
                if self.swap['target_slot'] is None:
                    slot_swap_entity = CsmSlotEntity(target_slot=self.name, preserve_vnet=self.swap['preserve_vnet'])
                    response = self.web_client.web_apps.begin_swap_slot_with_production(resource_group_name=self.resource_group, name=self.webapp_name, slot_swap_entity=slot_swap_entity)
                else:
                    slot_swap_entity = CsmSlotEntity(target_slot=self.swap['target_slot'], preserve_vnet=self.swap['preserve_vnet'])
                    response = self.web_client.web_apps.begin_swap_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name, slot_swap_entity=slot_swap_entity)
            elif self.swap['action'] == 'preview':
                if self.swap['target_slot'] is None:
                    slot_swap_entity = CsmSlotEntity(target_slot=self.name, preserve_vnet=self.swap['preserve_vnet'])
                    response = self.web_client.web_apps.apply_slot_config_to_production(resource_group_name=self.resource_group, name=self.webapp_name, slot_swap_entity=slot_swap_entity)
                else:
                    slot_swap_entity = CsmSlotEntity(target_slot=self.swap['target_slot'], preserve_vnet=self.swap['preserve_vnet'])
                    response = self.web_client.web_apps.apply_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name, slot_swap_entity=slot_swap_entity)
            elif self.swap['action'] == 'reset':
                if self.swap['target_slot'] is None:
                    response = self.web_client.web_apps.reset_production_slot_config(resource_group_name=self.resource_group, name=self.webapp_name)
                else:
                    response = self.web_client.web_apps.reset_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.swap['target_slot'])
                response = self.web_client.web_apps.reset_slot_configuration_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=self.name)
            self.log('Response : {0}'.format(response))
            return response
        except Exception as ex:
            self.fail('Failed to swap web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))

    def clone_slot(self):
        if self.configuration_source:
            src_slot = None if self.configuration_source.lower() == self.webapp_name.lower() else self.configuration_source
            if src_slot is None:
                site_config_clone_from = self.get_configuration()
            else:
                site_config_clone_from = self.get_configuration_slot(slot_name=src_slot)
            self.update_configuration_slot(site_config=site_config_clone_from)
            if src_slot is None:
                app_setting_clone_from = self.list_app_settings()
            else:
                app_setting_clone_from = self.list_app_settings_slot(src_slot)
            if self.app_settings:
                app_setting_clone_from.update(self.app_settings)
            self.update_app_settings_slot(app_settings=app_setting_clone_from)