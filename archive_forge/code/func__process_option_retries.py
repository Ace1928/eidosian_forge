from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def _process_option_retries(self):
    """check if retries option is int or dict and interpret it appropriately"""
    retries_opt = self._options.get_option('retries')
    if retries_opt is None:
        return
    retries = self._RETRIES_DEFAULT_PARAMS.copy()
    try:
        retries_int = check_type_int(retries_opt)
        if retries_int < 0:
            raise ValueError('Number of retries must be >= 0 (got %i)' % retries_int)
        elif retries_int == 0:
            retries = None
        else:
            retries['total'] = retries_int
    except TypeError:
        try:
            retries = check_type_dict(retries_opt)
        except TypeError:
            raise TypeError('retries option must be interpretable as int or dict. Got: %r' % retries_opt)
    self._options.set_option('retries', retries)