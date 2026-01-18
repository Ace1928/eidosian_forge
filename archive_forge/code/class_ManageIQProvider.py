from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQProvider(object):
    """
        Object to execute provider management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client

    def class_name_to_type(self, class_name):
        """ Convert class_name to type

        Returns:
            the type
        """
        out = [k for k, v in supported_providers().items() if v['class_name'] == class_name]
        if len(out) == 1:
            return out[0]
        return None

    def zone_id(self, name):
        """ Search for zone id by zone name.

        Returns:
            the zone id, or send a module Fail signal if zone not found.
        """
        zone = self.manageiq.find_collection_resource_by('zones', name=name)
        if not zone:
            self.module.fail_json(msg='zone %s does not exist in manageiq' % name)
        return zone['id']

    def provider(self, name):
        """ Search for provider object by name.

        Returns:
            the provider, or None if provider not found.
        """
        return self.manageiq.find_collection_resource_by('providers', name=name)

    def build_connection_configurations(self, provider_type, endpoints):
        """ Build "connection_configurations" objects from
        requested endpoints provided by user

        Returns:
            the user requested provider endpoints list
        """
        connection_configurations = []
        endpoint_keys = endpoint_list_spec().keys()
        provider_defaults = supported_providers().get(provider_type, {})
        endpoint = endpoints.get('provider')
        default_auth_key = endpoint.get('auth_key')
        for endpoint_key in endpoint_keys:
            endpoint = endpoints.get(endpoint_key)
            if endpoint:
                role = endpoint.get('role') or provider_defaults.get(endpoint_key + '_role', 'default')
                if role == 'default':
                    authtype = provider_defaults.get('authtype') or role
                else:
                    authtype = role
                connection_configurations.append({'endpoint': {'role': role, 'hostname': endpoint.get('hostname'), 'port': endpoint.get('port'), 'verify_ssl': [0, 1][endpoint.get('validate_certs', True)], 'security_protocol': endpoint.get('security_protocol'), 'certificate_authority': endpoint.get('certificate_authority'), 'path': endpoint.get('path')}, 'authentication': {'authtype': authtype, 'userid': endpoint.get('userid'), 'password': endpoint.get('password'), 'auth_key': endpoint.get('auth_key') or default_auth_key}})
        return connection_configurations

    def delete_provider(self, provider):
        """ Deletes a provider from manageiq.

        Returns:
            a short message describing the operation executed.
        """
        try:
            url = '%s/providers/%s' % (self.api_url, provider['id'])
            result = self.client.post(url, action='delete')
        except Exception as e:
            self.module.fail_json(msg='failed to delete provider %s: %s' % (provider['name'], str(e)))
        return dict(changed=True, msg=result['message'])

    def edit_provider(self, provider, name, provider_type, endpoints, zone_id, provider_region, host_default_vnc_port_start, host_default_vnc_port_end, subscription, project, uid_ems, tenant_mapping_enabled, api_version):
        """ Edit a provider from manageiq.

        Returns:
            a short message describing the operation executed.
        """
        url = '%s/providers/%s' % (self.api_url, provider['id'])
        resource = dict(name=name, zone={'id': zone_id}, provider_region=provider_region, connection_configurations=endpoints, host_default_vnc_port_start=host_default_vnc_port_start, host_default_vnc_port_end=host_default_vnc_port_end, subscription=subscription, project=project, uid_ems=uid_ems, tenant_mapping_enabled=tenant_mapping_enabled, api_version=api_version)
        resource = delete_nulls(resource)
        try:
            result = self.client.post(url, action='edit', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to update provider %s: %s' % (provider['name'], str(e)))
        return dict(changed=True, msg='successfully updated the provider %s: %s' % (provider['name'], result))

    def create_provider(self, name, provider_type, endpoints, zone_id, provider_region, host_default_vnc_port_start, host_default_vnc_port_end, subscription, project, uid_ems, tenant_mapping_enabled, api_version):
        """ Creates the provider in manageiq.

        Returns:
            a short message describing the operation executed.
        """
        resource = dict(name=name, zone={'id': zone_id}, provider_region=provider_region, host_default_vnc_port_start=host_default_vnc_port_start, host_default_vnc_port_end=host_default_vnc_port_end, subscription=subscription, project=project, uid_ems=uid_ems, tenant_mapping_enabled=tenant_mapping_enabled, api_version=api_version, connection_configurations=endpoints)
        resource = delete_nulls(resource)
        try:
            url = '%s/providers' % self.api_url
            result = self.client.post(url, type=supported_providers()[provider_type]['class_name'], **resource)
        except Exception as e:
            self.module.fail_json(msg='failed to create provider %s: %s' % (name, str(e)))
        return dict(changed=True, msg='successfully created the provider %s: %s' % (name, result['results']))

    def refresh(self, provider, name):
        """ Trigger provider refresh.

        Returns:
            a short message describing the operation executed.
        """
        try:
            url = '%s/providers/%s' % (self.api_url, provider['id'])
            result = self.client.post(url, action='refresh')
        except Exception as e:
            self.module.fail_json(msg='failed to refresh provider %s: %s' % (name, str(e)))
        return dict(changed=True, msg='refreshing provider %s' % name)