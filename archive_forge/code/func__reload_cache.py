from __future__ import (absolute_import, division, print_function)
import socket
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name
from ansible.module_utils.six import text_type
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _reload_cache(self):
    if self.get_option('cache_fallback'):
        self.display.vvv('Cannot connect to server, loading cache\n')
        self._options['cache_timeout'] = 0
        self.load_cache_plugin()
        self._cache.get(self.cache_key, {})