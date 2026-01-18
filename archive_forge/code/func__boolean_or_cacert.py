from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def _boolean_or_cacert(self):
    """return a bool or cacert"""
    ca_cert = self._options.get_option('ca_cert')
    validate_certs = self._options.get_option('validate_certs')
    if validate_certs is None:
        vault_skip_verify = os.environ.get('VAULT_SKIP_VERIFY')
        if vault_skip_verify is not None:
            try:
                vault_skip_verify = check_type_bool(vault_skip_verify)
            except TypeError:
                validate_certs = True
            else:
                validate_certs = not vault_skip_verify
        else:
            validate_certs = True
    if not (validate_certs and ca_cert):
        self._conopt_verify = validate_certs
    else:
        self._conopt_verify = to_text(ca_cert, errors='surrogate_or_strict')