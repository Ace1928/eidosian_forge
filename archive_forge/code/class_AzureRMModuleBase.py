from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
class AzureRMModuleBase(object):

    def __init__(self, derived_arg_spec, bypass_checks=False, no_log=False, check_invalid_arguments=None, mutually_exclusive=None, required_together=None, required_one_of=None, add_file_common_args=False, supports_check_mode=False, required_if=None, supports_tags=True, facts_module=False, skip_exec=False, is_ad_resource=False):
        merged_arg_spec = dict()
        merged_arg_spec.update(AZURE_COMMON_ARGS)
        if supports_tags:
            merged_arg_spec.update(AZURE_TAG_ARGS)
        if derived_arg_spec:
            merged_arg_spec.update(derived_arg_spec)
        merged_required_if = list(AZURE_COMMON_REQUIRED_IF)
        if required_if:
            merged_required_if += required_if
        self.module = AnsibleModule(argument_spec=merged_arg_spec, bypass_checks=bypass_checks, no_log=no_log, mutually_exclusive=mutually_exclusive, required_together=required_together, required_one_of=required_one_of, add_file_common_args=add_file_common_args, supports_check_mode=supports_check_mode, required_if=merged_required_if)
        if not HAS_PACKAGING_VERSION:
            self.fail(msg=missing_required_lib('packaging'), exception=HAS_PACKAGING_VERSION_EXC)
        if not HAS_AZURE:
            self.fail(msg=missing_required_lib('ansible[azure] (azure >= {0})'.format(AZURE_MIN_RELEASE)), exception=HAS_AZURE_EXC)
        self._authorization_client = None
        self._network_client = None
        self._storage_client = None
        self._subscription_client = None
        self._management_group_client = None
        self._resource_client = None
        self._compute_client = None
        self._image_client = None
        self._dns_client = None
        self._private_dns_client = None
        self._web_client = None
        self._marketplace_client = None
        self._sql_client = None
        self._mysql_client = None
        self._mariadb_client = None
        self._postgresql_client = None
        self._containerregistry_client = None
        self._containerinstance_client = None
        self._containerservice_client = None
        self._managedcluster_client = None
        self._traffic_manager_management_client = None
        self._monitor_autoscale_settings_client = None
        self._monitor_log_profiles_client = None
        self._monitor_diagnostic_settings_client = None
        self._resource = None
        self._log_analytics_client = None
        self._servicebus_client = None
        self._automation_client = None
        self._IoThub_client = None
        self._lock_client = None
        self._recovery_services_backup_client = None
        self._search_client = None
        self._datalake_store_client = None
        self._datafactory_client = None
        self._notification_hub_client = None
        self._event_hub_client = None
        self.check_mode = self.module.check_mode
        self.api_profile = self.module.params.get('api_profile')
        self.facts_module = facts_module
        self.azure_auth = AzureRMAuth(fail_impl=self.fail, is_ad_resource=is_ad_resource, **self.module.params)
        if self.module.params.get('tags'):
            self.validate_tags(self.module.params['tags'])
        if not skip_exec:
            res = self.exec_module(**self.module.params)
            self.module.exit_json(**res)

    def check_client_version(self, client_type):
        package_version = AZURE_PKG_VERSIONS.get(client_type.__name__, None)
        if package_version is not None:
            client_name = package_version.get('package_name')
            try:
                client_module = importlib.import_module(client_type.__module__)
                client_version = client_module.VERSION
            except (RuntimeError, AttributeError):
                return
            expected_version = package_version.get('expected_version')
            if Version(client_version) < Version(expected_version):
                self.fail('Installed azure-mgmt-{0} client version is {1}. The minimum supported version is {2}. Try `pip install ansible[azure]`'.format(client_name, client_version, expected_version))
            if Version(client_version) != Version(expected_version):
                self.module.warn('Installed azure-mgmt-{0} client version is {1}. The expected version is {2}. Try `pip install ansible[azure]`'.format(client_name, client_version, expected_version))

    def exec_module(self, **kwargs):
        self.fail('Error: {0} failed to implement exec_module method.'.format(self.__class__.__name__))

    def fail(self, msg, **kwargs):
        """
        Shortcut for calling module.fail()

        :param msg: Error message text.
        :param kwargs: Any key=value pairs
        :return: None
        """
        self.module.fail_json(msg=msg, **kwargs)

    def deprecate(self, msg, version=None):
        self.module.deprecate(msg, version)

    def log(self, msg, pretty_print=False):
        if pretty_print:
            self.module.debug(json.dumps(msg, indent=4, sort_keys=True))
        else:
            self.module.debug(msg)

    def validate_tags(self, tags):
        """
        Check if tags dictionary contains string:string pairs.

        :param tags: dictionary of string:string pairs
        :return: None
        """
        if not self.facts_module:
            if not isinstance(tags, dict):
                self.fail('Tags must be a dictionary of string:string values.')
            for key, value in tags.items():
                if not isinstance(value, str):
                    self.fail('Tags values must be strings. Found {0}:{1}'.format(str(key), str(value)))

    def update_tags(self, tags):
        """
        Call from the module to update metadata tags. Returns tuple
        with bool indicating if there was a change and dict of new
        tags to assign to the object.

        :param tags: metadata tags from the object
        :return: bool, dict
        """
        tags = tags or dict()
        new_tags = copy.copy(tags) if isinstance(tags, dict) else dict()
        param_tags = self.module.params.get('tags') if isinstance(self.module.params.get('tags'), dict) else dict()
        append_tags = self.module.params.get('append_tags') if self.module.params.get('append_tags') is not None else True
        changed = False
        for key, value in param_tags.items():
            if not new_tags.get(key) or new_tags[key] != value:
                changed = True
                new_tags[key] = value
        if not append_tags:
            for key, value in tags.items():
                if not param_tags.get(key):
                    new_tags.pop(key)
                    changed = True
        return (changed, new_tags)

    def has_tags(self, obj_tags, tag_list):
        """
        Used in fact modules to compare object tags to list of parameter tags. Return true if list of parameter tags
        exists in object tags.

        :param obj_tags: dictionary of tags from an Azure object.
        :param tag_list: list of tag keys or tag key:value pairs
        :return: bool
        """
        if not obj_tags and tag_list:
            return False
        if not tag_list:
            return True
        matches = 0
        result = False
        for tag in tag_list:
            tag_key = tag
            tag_value = None
            if ':' in tag:
                tag_key, tag_value = tag.split(':')
            if tag_value and obj_tags.get(tag_key) == tag_value:
                matches += 1
            elif not tag_value and obj_tags.get(tag_key):
                matches += 1
        if matches == len(tag_list):
            result = True
        return result

    def get_resource_group(self, resource_group):
        """
        Fetch a resource group.

        :param resource_group: name of a resource group
        :return: resource group object
        """
        try:
            return self.rm_client.resource_groups.get(resource_group)
        except Exception as exc:
            self.fail('Error retrieving resource group {0} - {1}'.format(resource_group, str(exc)))

    def parse_resource_to_dict(self, resource):
        """
        Return a dict of the give resource, which contains name and resource group.

        :param resource: It can be a resource name, id or a dict contains name and resource group.
        """
        resource_dict = parse_resource_id(resource) if not isinstance(resource, dict) else resource
        resource_dict['resource_group'] = resource_dict.get('resource_group', self.resource_group)
        resource_dict['subscription_id'] = resource_dict.get('subscription_id', self.subscription_id)
        return resource_dict

    def serialize_obj(self, obj, class_name, enum_modules=None):
        """
        Return a JSON representation of an Azure object.

        :param obj: Azure object
        :param class_name: Name of the object's class
        :param enum_modules: List of module names to build enum dependencies from.
        :return: serialized result
        """
        return obj.as_dict()

    def get_poller_result(self, poller, wait=5):
        """
        Consistent method of waiting on and retrieving results from Azure's long poller

        :param poller Azure poller object
        :return object resulting from the original request
        """
        try:
            delay = wait
            while not poller.done():
                self.log('Waiting for {0} sec'.format(delay))
                poller.wait(timeout=delay)
            return poller.result()
        except Exception as exc:
            self.log(str(exc))
            raise

    def get_multiple_pollers_results(self, pollers, wait=0.05):
        """
        Consistent method of waiting on and retrieving results from multiple Azure's long poller

        :param pollers list of Azure poller object
        :param wait Period of time to wait for the long running operation to complete.
        :return list of object resulting from the original request
        """

        def _continue_polling():
            return not all((poller.done() for poller in pollers))
        try:
            while _continue_polling():
                for poller in pollers:
                    if poller.done():
                        continue
                    self.log('Waiting for {0} sec'.format(wait))
                    poller.wait(timeout=wait)
            return [poller.result() for poller in pollers]
        except Exception as exc:
            self.log(str(exc))
            raise

    def check_provisioning_state(self, azure_object, requested_state='present'):
        """
        Check an Azure object's provisioning state. If something did not complete the provisioning
        process, then we cannot operate on it.

        :param azure_object An object such as a subnet, storageaccount, etc. Must have provisioning_state
                            and name attributes.
        :return None
        """
        if hasattr(azure_object, 'properties') and hasattr(azure_object.properties, 'provisioning_state') and hasattr(azure_object, 'name'):
            if isinstance(azure_object.properties.provisioning_state, Enum):
                if azure_object.properties.provisioning_state.value != AZURE_SUCCESS_STATE and requested_state != 'absent':
                    self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.properties.provisioning_state, AZURE_SUCCESS_STATE))
                return
            if azure_object.properties.provisioning_state != AZURE_SUCCESS_STATE and requested_state != 'absent':
                self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.properties.provisioning_state, AZURE_SUCCESS_STATE))
            return
        if hasattr(azure_object, 'provisioning_state') or not hasattr(azure_object, 'name'):
            if isinstance(azure_object.provisioning_state, Enum):
                if azure_object.provisioning_state.value != AZURE_SUCCESS_STATE and requested_state != 'absent':
                    self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.provisioning_state, AZURE_SUCCESS_STATE))
                return
            if azure_object.provisioning_state != AZURE_SUCCESS_STATE and requested_state != 'absent':
                self.fail('Error {0} has a provisioning state of {1}. Expecting state to be {2}.'.format(azure_object.name, azure_object.provisioning_state, AZURE_SUCCESS_STATE))

    def get_blob_service_client(self, resource_group_name, storage_account_name):
        try:
            self.log('Getting storage account detail')
            account = self.storage_client.storage_accounts.get_properties(resource_group_name=resource_group_name, account_name=storage_account_name)
            account_keys = self.storage_client.storage_accounts.list_keys(resource_group_name=resource_group_name, account_name=storage_account_name)
        except Exception as exc:
            self.fail('Error getting storage account detail for {0}: {1}'.format(storage_account_name, str(exc)))
        try:
            self.log('Create blob service client')
            return BlobServiceClient(account_url=account.primary_endpoints.blob, credential=account_keys.keys[0].value)
        except Exception as exc:
            self.fail('Error creating blob service client for storage account {0} - {1}'.format(storage_account_name, str(exc)))

    def create_default_pip(self, resource_group, location, public_ip_name, allocation_method='Dynamic', sku=None):
        """
        Create a default public IP address <public_ip_name> to associate with a network interface.
        If a PIP address matching <public_ip_name> exists, return it. Otherwise, create one.

        :param resource_group: name of an existing resource group
        :param location: a valid azure location
        :param public_ip_name: base name to assign the public IP address
        :param allocation_method: one of 'Static' or 'Dynamic'
        :param sku: sku
        :return: PIP object
        """
        pip = None
        self.log('Starting create_default_pip {0}'.format(public_ip_name))
        self.log('Check to see if public IP {0} exists'.format(public_ip_name))
        try:
            pip = self.network_client.public_ip_addresses.get(resource_group, public_ip_name)
        except Exception:
            pass
        if pip:
            self.log('Public ip {0} found.'.format(public_ip_name))
            self.check_provisioning_state(pip)
            return pip
        params = self.network_models.PublicIPAddress(location=location, public_ip_allocation_method=allocation_method, sku=sku)
        self.log('Creating default public IP {0}'.format(public_ip_name))
        try:
            poller = self.network_client.public_ip_addresses.begin_create_or_update(resource_group, public_ip_name, params)
        except Exception as exc:
            self.fail('Error creating {0} - {1}'.format(public_ip_name, str(exc)))
        return self.get_poller_result(poller)

    def create_default_securitygroup(self, resource_group, location, security_group_name, os_type, open_ports):
        """
        Create a default security group <security_group_name> to associate with a network interface. If a security group matching
        <security_group_name> exists, return it. Otherwise, create one.

        :param resource_group: Resource group name
        :param location: azure location name
        :param security_group_name: base name to use for the security group
        :param os_type: one of 'Windows' or 'Linux'. Determins any default rules added to the security group.
        :param ssh_port: for os_type 'Linux' port used in rule allowing SSH access.
        :param rdp_port: for os_type 'Windows' port used in rule allowing RDP access.
        :return: security_group object
        """
        group = None
        self.log('Create security group {0}'.format(security_group_name))
        self.log('Check to see if security group {0} exists'.format(security_group_name))
        try:
            group = self.network_client.network_security_groups.get(resource_group, security_group_name)
        except Exception:
            pass
        if group:
            self.log('Security group {0} found.'.format(security_group_name))
            self.check_provisioning_state(group)
            return group
        parameters = self.network_models.NetworkSecurityGroup()
        parameters.location = location
        if not open_ports:
            if os_type == 'Linux':
                parameters.security_rules = [self.network_models.SecurityRule(protocol='Tcp', source_address_prefix='*', destination_address_prefix='*', access='Allow', direction='Inbound', description='Allow SSH Access', source_port_range='*', destination_port_range='22', priority=100, name='SSH')]
                parameters.location = location
            else:
                parameters.security_rules = [self.network_models.SecurityRule(protocol='Tcp', source_address_prefix='*', destination_address_prefix='*', access='Allow', direction='Inbound', description='Allow RDP port 3389', source_port_range='*', destination_port_range='3389', priority=100, name='RDP01'), self.network_models.SecurityRule(protocol='Tcp', source_address_prefix='*', destination_address_prefix='*', access='Allow', direction='Inbound', description='Allow WinRM HTTPS port 5986', source_port_range='*', destination_port_range='5986', priority=101, name='WinRM01')]
        else:
            parameters.security_rules = []
            priority = 100
            for port in open_ports:
                priority += 1
                rule_name = 'Rule_{0}'.format(priority)
                parameters.security_rules.append(self.network_models.SecurityRule(protocol='Tcp', source_address_prefix='*', destination_address_prefix='*', access='Allow', direction='Inbound', source_port_range='*', destination_port_range=str(port), priority=priority, name=rule_name))
        self.log('Creating default security group {0}'.format(security_group_name))
        try:
            poller = self.network_client.network_security_groups.begin_create_or_update(resource_group, security_group_name, parameters)
        except Exception as exc:
            self.fail('Error creating default security rule {0} - {1}'.format(security_group_name, str(exc)))
        return self.get_poller_result(poller)

    @staticmethod
    def _validation_ignore_callback(session, global_config, local_config, **kwargs):
        session.verify = False

    def get_api_profile(self, client_type_name, api_profile_name):
        profile_all_clients = AZURE_API_PROFILES.get(api_profile_name)
        if not profile_all_clients:
            raise KeyError('unknown Azure API profile: {0}'.format(api_profile_name))
        profile_raw = profile_all_clients.get(client_type_name, None)
        if not profile_raw:
            self.module.warn('Azure API profile {0} does not define an entry for {1}'.format(api_profile_name, client_type_name))
        if isinstance(profile_raw, dict):
            if not profile_raw.get('default_api_version'):
                raise KeyError("Azure API profile {0} does not define 'default_api_version'".format(api_profile_name))
            return profile_raw
        return dict(default_api_version=profile_raw)

    def get_graphrbac_client(self, tenant_id):
        cred = self.azure_auth.azure_credentials
        base_url = self.azure_auth._cloud_environment.endpoints.active_directory_graph_resource_id
        client = GraphRbacManagementClient(cred, tenant_id, base_url)
        return client

    def get_mgmt_svc_client(self, client_type, base_url=None, api_version=None, suppress_subscription_id=False, is_track2=False):
        self.log('Getting management service client {0}'.format(client_type.__name__))
        self.check_client_version(client_type)
        client_argspec = inspect.signature(client_type.__init__)
        if not base_url:
            base_url = self.azure_auth._cloud_environment.endpoints.resource_manager
        if not base_url.endswith('/'):
            base_url += '/'
        mgmt_subscription_id = self.azure_auth.subscription_id
        if self.module.params.get('subscription_id'):
            mgmt_subscription_id = self.module.params.get('subscription_id')
        if suppress_subscription_id:
            if is_track2:
                client_kwargs = dict(credential=self.azure_auth.azure_credential_track2, base_url=base_url, credential_scopes=[base_url + '.default'])
            else:
                client_kwargs = dict(credentials=self.azure_auth.azure_credentials, base_url=base_url)
        elif is_track2:
            client_kwargs = dict(credential=self.azure_auth.azure_credential_track2, subscription_id=mgmt_subscription_id, base_url=base_url, credential_scopes=[base_url + '.default'])
        else:
            client_kwargs = dict(credentials=self.azure_auth.azure_credentials, subscription_id=mgmt_subscription_id, base_url=base_url)
        api_profile_dict = {}
        if self.api_profile:
            api_profile_dict = self.get_api_profile(client_type.__name__, self.api_profile)
        if api_profile_dict and 'profile' in client_argspec.parameters:
            client_kwargs['profile'] = api_profile_dict
        if 'api_version' in client_argspec.parameters:
            profile_default_version = api_profile_dict.get('default_api_version', None)
            if api_version or profile_default_version:
                client_kwargs['api_version'] = api_version or profile_default_version
                if 'profile' in client_kwargs:
                    client_kwargs.pop('profile')
        client = client_type(**client_kwargs)
        try:
            getattr(client, 'models')
        except AttributeError:

            def _ansible_get_models(self, *arg, **kwarg):
                return self._ansible_models
            setattr(client, '_ansible_models', importlib.import_module(client_type.__module__).models)
            client.models = types.MethodType(_ansible_get_models, client)
        if not is_track2:
            client.config = self.add_user_agent(client.config)
            if self.azure_auth._cert_validation_mode == 'ignore':
                client.config.session_configuration_callback = self._validation_ignore_callback
        elif self.azure_auth._cert_validation_mode == 'ignore':
            client._config.session_configuration_callback = self._validation_ignore_callback
        return client

    def add_user_agent(self, config):
        config.add_user_agent(ANSIBLE_USER_AGENT)
        if CLOUDSHELL_USER_AGENT_KEY in os.environ:
            config.add_user_agent(os.environ[CLOUDSHELL_USER_AGENT_KEY])
        if VSCODEEXT_USER_AGENT_KEY in os.environ:
            config.add_user_agent(os.environ[VSCODEEXT_USER_AGENT_KEY])
        return config

    def generate_sas_token(self, **kwags):
        base_url = kwags.get('base_url', None)
        expiry = kwags.get('expiry', time() + 3600)
        key = kwags.get('key', None)
        policy = kwags.get('policy', None)
        url = quote_plus(base_url)
        ttl = int(expiry)
        sign_key = '{0}\n{1}'.format(url, ttl)
        signature = b64encode(HMAC(b64decode(key), sign_key.encode('utf-8'), sha256).digest())
        result = {'sr': url, 'sig': signature, 'se': str(ttl)}
        if policy:
            result['skn'] = policy
        return 'SharedAccessSignature ' + urlencode(result)

    def get_subnet_detail(self, subnet_id):
        vnet_detail = subnet_id.split('/Microsoft.Network/virtualNetworks/')[1].split('/subnets/')
        return dict(resource_group=subnet_id.split('resourceGroups/')[1].split('/')[0], vnet_name=vnet_detail[0], subnet_name=vnet_detail[1])

    @property
    def credentials(self):
        return self.azure_auth.credentials

    @property
    def _cloud_environment(self):
        return self.azure_auth._cloud_environment

    @property
    def subscription_id(self):
        return self.azure_auth.subscription_id

    @property
    def storage_client(self):
        self.log('Getting storage client...')
        if not self._storage_client:
            self._storage_client = self.get_mgmt_svc_client(StorageManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-06-01')
        return self._storage_client

    @property
    def storage_models(self):
        return StorageManagementClient.models('2021-06-01')

    @property
    def authorization_client(self):
        self.log('Getting authorization client...')
        if not self._authorization_client:
            self._authorization_client = self.get_mgmt_svc_client(AuthorizationManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2020-04-01-preview')
        return self._authorization_client

    @property
    def authorization_models(self):
        return AuthorizationManagementClient.models('2020-04-01-preview')

    @property
    def subscription_client(self):
        self.log('Getting subscription client...')
        if not self._subscription_client:
            self._subscription_client = self.get_mgmt_svc_client(SubscriptionClient, base_url=self._cloud_environment.endpoints.resource_manager, suppress_subscription_id=True, is_track2=True, api_version='2019-11-01')
        return self._subscription_client

    @property
    def subscription_models(self):
        return SubscriptionClient.models('2019-11-01')

    @property
    def management_groups_client(self):
        self.log('Getting Management Groups client...')
        if not self._management_group_client:
            self._management_group_client = self.get_mgmt_svc_client(ManagementGroupsClient, base_url=self._cloud_environment.endpoints.resource_manager, suppress_subscription_id=True, is_track2=True, api_version='2020-05-01')
        return self._management_group_client

    @property
    def network_client(self):
        self.log('Getting network client')
        if not self._network_client:
            self._network_client = self.get_mgmt_svc_client(NetworkManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-03-01')
        return self._network_client

    @property
    def network_models(self):
        self.log('Getting network models...')
        return NetworkManagementClient.models('2021-03-01')

    @property
    def rm_client(self):
        self.log('Getting resource manager client')
        if not self._resource_client:
            self._resource_client = self.get_mgmt_svc_client(ResourceManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2019-10-01')
        return self._resource_client

    @property
    def rm_models(self):
        self.log('Getting resource manager models')
        return ResourceManagementClient.models('2019-10-01')

    @property
    def image_client(self):
        self.log('Getting compute image client')
        if not self._image_client:
            self._image_client = self.get_mgmt_svc_client(ComputeManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-04-01')
        return self._image_client

    @property
    def image_models(self):
        self.log('Getting compute image models')
        return ComputeManagementClient.models('2021-04-01')

    @property
    def compute_client(self):
        self.log('Getting compute client')
        if not self._compute_client:
            self._compute_client = self.get_mgmt_svc_client(ComputeManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-04-01')
        return self._compute_client

    @property
    def compute_models(self):
        self.log('Getting compute models')
        return ComputeManagementClient.models('2021-04-01')

    @property
    def dns_client(self):
        self.log('Getting dns client')
        if not self._dns_client:
            self._dns_client = self.get_mgmt_svc_client(DnsManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2018-05-01')
        return self._dns_client

    @property
    def dns_models(self):
        self.log('Getting dns models...')
        return DnsManagementClient.models('2018-05-01')

    @property
    def private_dns_client(self):
        self.log('Getting private dns client')
        if not self._private_dns_client:
            self._private_dns_client = self.get_mgmt_svc_client(PrivateDnsManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._private_dns_client

    @property
    def private_dns_models(self):
        self.log('Getting private dns models')
        return PrivateDnsModels

    @property
    def web_client(self):
        self.log('Getting web client')
        if not self._web_client:
            self._web_client = self.get_mgmt_svc_client(WebSiteManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-03-01')
        return self._web_client

    @property
    def containerservice_client(self):
        self.log('Getting container service client')
        if not self._containerservice_client:
            self._containerservice_client = self.get_mgmt_svc_client(ContainerServiceClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2017-07-01')
        return self._containerservice_client

    @property
    def managedcluster_models(self):
        self.log('Getting container service models')
        return ContainerServiceClient.models('2022-02-01')

    @property
    def managedcluster_client(self):
        self.log('Getting container service client')
        if not self._managedcluster_client:
            self._managedcluster_client = self.get_mgmt_svc_client(ContainerServiceClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2022-02-01')
        return self._managedcluster_client

    @property
    def sql_client(self):
        self.log('Getting SQL client')
        if not self._sql_client:
            self._sql_client = self.get_mgmt_svc_client(SqlManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True)
        return self._sql_client

    @property
    def postgresql_client(self):
        self.log('Getting PostgreSQL client')
        if not self._postgresql_client:
            self._postgresql_client = self.get_mgmt_svc_client(PostgreSQLManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._postgresql_client

    @property
    def mysql_client(self):
        self.log('Getting MySQL client')
        if not self._mysql_client:
            self._mysql_client = self.get_mgmt_svc_client(MySQLManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._mysql_client

    @property
    def mariadb_client(self):
        self.log('Getting MariaDB client')
        if not self._mariadb_client:
            self._mariadb_client = self.get_mgmt_svc_client(MariaDBManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._mariadb_client

    @property
    def containerregistry_client(self):
        self.log('Getting container registry mgmt client')
        if not self._containerregistry_client:
            self._containerregistry_client = self.get_mgmt_svc_client(ContainerRegistryManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-09-01')
        return self._containerregistry_client

    @property
    def containerinstance_client(self):
        self.log('Getting container instance mgmt client')
        if not self._containerinstance_client:
            self._containerinstance_client = self.get_mgmt_svc_client(ContainerInstanceManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2018-06-01')
        return self._containerinstance_client

    @property
    def marketplace_client(self):
        self.log('Getting marketplace agreement client')
        if not self._marketplace_client:
            self._marketplace_client = self.get_mgmt_svc_client(MarketplaceOrderingAgreements, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._marketplace_client

    @property
    def traffic_manager_management_client(self):
        self.log('Getting traffic manager client')
        if not self._traffic_manager_management_client:
            self._traffic_manager_management_client = self.get_mgmt_svc_client(TrafficManagerManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._traffic_manager_management_client

    @property
    def monitor_autoscale_settings_client(self):
        self.log('Getting monitor client for autoscale_settings')
        if not self._monitor_autoscale_settings_client:
            self._monitor_autoscale_settings_client = self.get_mgmt_svc_client(MonitorManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2015-04-01', is_track2=True)
        return self._monitor_autoscale_settings_client

    @property
    def monitor_log_profiles_client(self):
        self.log('Getting monitor client for log_profiles')
        if not self._monitor_log_profiles_client:
            self._monitor_log_profiles_client = self.get_mgmt_svc_client(MonitorManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2016-03-01', is_track2=True)
        return self._monitor_log_profiles_client

    @property
    def monitor_diagnostic_settings_client(self):
        self.log('Getting monitor client for diagnostic_settings')
        if not self._monitor_diagnostic_settings_client:
            self._monitor_diagnostic_settings_client = self.get_mgmt_svc_client(MonitorManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, api_version='2021-05-01-preview', is_track2=True)
        return self._monitor_diagnostic_settings_client

    @property
    def log_analytics_client(self):
        self.log('Getting log analytics client')
        if not self._log_analytics_client:
            self._log_analytics_client = self.get_mgmt_svc_client(LogAnalyticsManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._log_analytics_client

    @property
    def log_analytics_models(self):
        self.log('Getting log analytics models')
        return LogAnalyticsModels

    @property
    def servicebus_client(self):
        self.log('Getting servicebus client')
        if not self._servicebus_client:
            self._servicebus_client = self.get_mgmt_svc_client(ServiceBusManagementClient, is_track2=True, api_version='2021-06-01-preview', base_url=self._cloud_environment.endpoints.resource_manager)
        return self._servicebus_client

    @property
    def servicebus_models(self):
        return ServiceBusManagementClient.models('2021-06-01-preview')

    @property
    def automation_client(self):
        self.log('Getting automation client')
        if not self._automation_client:
            self._automation_client = self.get_mgmt_svc_client(AutomationClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True)
        return self._automation_client

    @property
    def automation_models(self):
        return AutomationModel

    @property
    def IoThub_client(self):
        self.log('Getting iothub client')
        if not self._IoThub_client:
            self._IoThub_client = self.get_mgmt_svc_client(IotHubClient, is_track2=True, api_version='2018-04-01', base_url=self._cloud_environment.endpoints.resource_manager)
        return self._IoThub_client

    @property
    def IoThub_models(self):
        return IoTHubModels

    @property
    def lock_client(self):
        self.log('Getting lock client')
        if not self._lock_client:
            self._lock_client = self.get_mgmt_svc_client(ManagementLockClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2016-09-01')
        return self._lock_client

    @property
    def lock_models(self):
        self.log('Getting lock models')
        return ManagementLockClient.models('2016-09-01')

    @property
    def recovery_services_backup_client(self):
        self.log('Getting recovery services backup client')
        if not self._recovery_services_backup_client:
            self._recovery_services_backup_client = self.get_mgmt_svc_client(RecoveryServicesBackupClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._recovery_services_backup_client

    @property
    def recovery_services_backup_models(self):
        return RecoveryServicesBackupModels

    @property
    def search_client(self):
        self.log('Getting search client...')
        if not self._search_client:
            self._search_client = self.get_mgmt_svc_client(SearchManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2020-08-01')
        return self._search_client

    @property
    def datalake_store_client(self):
        self.log('Getting datalake store client...')
        if not self._datalake_store_client:
            self._datalake_store_client = self.get_mgmt_svc_client(DataLakeStoreAccountManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2016-11-01')
        return self._datalake_store_client

    @property
    def datalake_store_models(self):
        return DataLakeStoreAccountModel

    @property
    def notification_hub_client(self):
        self.log('Getting notification hub client')
        if not self._notification_hub_client:
            self._notification_hub_client = self.get_mgmt_svc_client(NotificationHubsManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2016-03-01')
        return self._notification_hub_client

    @property
    def event_hub_client(self):
        self.log('Getting event hub client')
        if not self._event_hub_client:
            self._event_hub_client = self.get_mgmt_svc_client(EventHubManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-11-01')
        return self._event_hub_client

    @property
    def datafactory_client(self):
        self.log('Getting datafactory client...')
        if not self._datafactory_client:
            self._datafactory_client = self.get_mgmt_svc_client(DataFactoryManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        return self._datafactory_client

    @property
    def datafactory_model(self):
        return DataFactoryModel