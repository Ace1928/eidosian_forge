from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
from ansible_collections.community.hashi_vault.plugins.module_utils._connection_options import HashiVaultConnectionOptions
from ansible_collections.community.hashi_vault.plugins.module_utils._authenticator import HashiVaultAuthenticator
class HashiVaultModule(AnsibleModule):

    def __init__(self, *args, **kwargs):
        if 'hashi_vault_custom_retry_callback' in kwargs:
            callback = kwargs.pop('hashi_vault_custom_retry_callback')
        else:
            callback = self._generate_retry_callback
        super(HashiVaultModule, self).__init__(*args, **kwargs)
        self.helper = HashiVaultHelper()
        self.adapter = HashiVaultOptionAdapter.from_dict(self.params)
        self.connection_options = HashiVaultConnectionOptions(option_adapter=self.adapter, retry_callback_generator=callback)
        self.authenticator = HashiVaultAuthenticator(option_adapter=self.adapter, warning_callback=self.warn, deprecate_callback=self.deprecate)

    @classmethod
    def generate_argspec(cls, **kwargs):
        spec = HashiVaultConnectionOptions.ARGSPEC.copy()
        spec.update(HashiVaultAuthenticator.ARGSPEC.copy())
        spec.update(**kwargs)
        return spec

    def _generate_retry_callback(self, retry_action):
        """returns a Retry callback function for modules"""

        def _on_retry(retry_obj):
            if retry_obj.total > 0:
                if retry_action == 'warn':
                    self.warn('community.hashi_vault: %i %s remaining.' % (retry_obj.total, 'retry' if retry_obj.total == 1 else 'retries'))
                else:
                    pass
        return _on_retry