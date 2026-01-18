from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMKeyVaultKey(AzureRMModuleBase):
    """ Module that creates or deletes keys in Azure KeyVault """

    def __init__(self):
        self.module_arg_spec = dict(key_name=dict(type='str', required=True), keyvault_uri=dict(type='str', no_log=True, required=True), key_type=dict(type='str', default='RSA'), key_size=dict(type='int'), key_attributes=dict(type='dict', no_log=True, options=key_addribute_spec), curve=dict(type='str'), pem_file=dict(type='str'), pem_password=dict(type='str', no_log=True), byok_file=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.results = dict(changed=False, state=dict())
        self.key_name = None
        self.keyvault_uri = None
        self.key_type = None
        self.key_size = None
        self.key_attributes = None
        self.curve = None
        self.pem_file = None
        self.pem_password = None
        self.state = None
        self.client = None
        self.tags = None
        required_if = [('pem_password', 'present', ['pem_file'])]
        super(AzureRMKeyVaultKey, self).__init__(self.module_arg_spec, supports_check_mode=True, required_if=required_if, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.client = self.get_keyvault_client()
        results = dict()
        changed = False
        try:
            results['key_id'] = self.get_key(self.key_name)
            if self.state == 'absent':
                changed = True
        except Exception:
            if self.state == 'present':
                changed = True
        self.results['changed'] = changed
        self.results['state'] = results
        if not self.check_mode:
            if self.state == 'present' and changed:
                results['key_id'] = self.create_key(self.key_name)
                self.results['state'] = results
                self.results['state']['status'] = 'Created'
            elif self.state == 'absent' and changed:
                results['key_id'] = self.delete_key(self.key_name)
                self.results['state'] = results
                self.results['state']['status'] = 'Deleted'
        elif self.state == 'present' and changed:
            self.results['state']['status'] = 'Created'
        elif self.state == 'absent' and changed:
            self.results['state']['status'] = 'Deleted'
        return self.results

    def get_keyvault_client(self):
        return KeyClient(vault_url=self.keyvault_uri, credential=self.azure_auth.azure_credential_track2)

    def get_key(self, name, version=''):
        """ Gets an existing key """
        key_bundle = self.client.get_key(name, version)
        if key_bundle:
            return key_bundle.id

    def create_key(self, name):
        """ Creates a key """
        if self.key_attributes is not None:
            k_enabled = self.key_attributes.get('enabled', True)
            k_not_before = self.key_attributes.get('not_before', None)
            k_expires = self.key_attributes.get('expires', None)
            if k_not_before:
                k_not_before = datetime.fromisoformat(k_not_before.replace('Z', '+00:00'))
            if k_expires:
                k_expires = datetime.fromisoformat(k_expires.replace('Z', '+00:00'))
        else:
            k_enabled = True
            k_not_before = None
            k_expires = None
        key_bundle = self.client.create_key(name=name, key_type=self.key_type, size=self.key_size, curve=self.curve, tags=self.tags, enabled=k_enabled, not_before=k_not_before, expires_on=k_expires)
        return key_bundle._properties._id

    def delete_key(self, name):
        """ Deletes a key """
        deleted_key = self.client.begin_delete_key(name)
        result = self.get_poller_result(deleted_key)
        return result.properties._id