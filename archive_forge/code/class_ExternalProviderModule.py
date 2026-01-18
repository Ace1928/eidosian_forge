from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class ExternalProviderModule(BaseModule):
    non_provider_params = ['type', 'authentication_keys', 'data_center']

    def provider_type(self, provider_type):
        self._provider_type = provider_type

    def provider_module_params(self):
        provider_params = [(key, value) for key, value in self._module.params.items() if key not in self.non_provider_params]
        provider_params.append(('data_center', self.get_data_center()))
        return provider_params

    def get_data_center(self):
        dc_name = self._module.params.get('data_center', None)
        if dc_name:
            system_service = self._connection.system_service()
            data_centers_service = system_service.data_centers_service()
            return data_centers_service.list(search='name=%s' % dc_name)[0]
        return dc_name

    def build_entity(self):
        provider_type = self._provider_type(requires_authentication=self._module.params.get('username') is not None)
        if self._module.params.pop('type') == NETWORK:
            setattr(provider_type, 'type', otypes.OpenStackNetworkProviderType(self._module.params.pop('network_type')))
        for key, value in self.provider_module_params():
            if hasattr(provider_type, key):
                setattr(provider_type, key, value)
        return provider_type

    def update_check(self, entity):
        return equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('url'), entity.url) and equal(self._module.params.get('authentication_url'), entity.authentication_url) and equal(self._module.params.get('tenant_name'), getattr(entity, 'tenant_name', None)) and equal(self._module.params.get('username'), entity.username)

    def update_volume_provider_auth_keys(self, provider, providers_service, keys):
        """
        Update auth keys for volume provider, if not exist add them or remove
        if they are not specified and there are already defined in the external
        volume provider.

        Args:
            provider (dict): Volume provider details.
            providers_service (openstack_volume_providers_service): Provider
                service.
            keys (list): Keys to be updated/added to volume provider, each key
                is represented as dict with keys: uuid, value.
        """
        provider_service = providers_service.provider_service(provider['id'])
        auth_keys_service = provider_service.authentication_keys_service()
        provider_keys = auth_keys_service.list()
        for key in [k.id for k in provider_keys if k.uuid not in [defined_key['uuid'] for defined_key in keys]]:
            self.changed = True
            if not self._module.check_mode:
                auth_keys_service.key_service(key).remove()
        if not (provider_keys or keys):
            return
        for key in keys:
            key_id_for_update = None
            for existing_key in provider_keys:
                if key['uuid'] == existing_key.uuid:
                    key_id_for_update = existing_key.id
            auth_key_usage_type = otypes.OpenstackVolumeAuthenticationKeyUsageType('ceph')
            auth_key = otypes.OpenstackVolumeAuthenticationKey(usage_type=auth_key_usage_type, uuid=key['uuid'], value=key['value'])
            if not key_id_for_update:
                self.changed = True
                if not self._module.check_mode:
                    auth_keys_service.add(auth_key)
            else:
                self.changed = True
                if not self._module.check_mode:
                    auth_key_service = auth_keys_service.key_service(key_id_for_update)
                    auth_key_service.update(auth_key)