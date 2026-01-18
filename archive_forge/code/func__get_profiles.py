from __future__ import (absolute_import, division, print_function)
import socket
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name
from ansible.module_utils.six import text_type
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _get_profiles(self):
    if not self.use_cache or 'profiles' not in self._cache.get(self.cache_key, {}):
        c = self._get_connection()
        try:
            if self.token is not None:
                data = c.get_profiles(self.token)
            else:
                data = c.get_profiles()
        except (socket.gaierror, socket.error, xmlrpc_client.ProtocolError):
            self._reload_cache()
        else:
            self._init_cache()
            self._cache[self.cache_key]['profiles'] = data
    return self._cache[self.cache_key]['profiles']