from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def _get_custom_requests_session(self, **retry_kwargs):
    """returns a requests.Session to pass to hvac (or None)"""
    if not HAS_RETRIES:
        raise NotImplementedError('Retries are unavailable. This may indicate very old versions of one or more of the following: hvac, requests, urllib3.')

    class CallbackRetry(urllib3.util.Retry):

        def __init__(self, *args, **kwargs):
            self._newcb = kwargs.pop('new_callback')
            super(CallbackRetry, self).__init__(*args, **kwargs)

        def new(self, **kwargs):
            if self._newcb is not None:
                self._newcb(self)
            kwargs['new_callback'] = self._newcb
            return super(CallbackRetry, self).new(**kwargs)
    if 'raise_on_status' not in retry_kwargs:
        retry_kwargs['raise_on_status'] = False
    retry = CallbackRetry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    sess = Session()
    sess.mount('https://', adapter)
    sess.mount('http://', adapter)
    return sess