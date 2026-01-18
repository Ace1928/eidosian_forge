from __future__ import (absolute_import, division, print_function)
import itertools
import re
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _can_add_host(self, name, properties):
    """Ensure that a host satisfies all defined hosts filters. If strict mode is
        enabled, any error during host filter compositing will lead to an AnsibleError
        being raised, otherwise the filter will be ignored.
        """
    for host_filter in self.host_filters:
        try:
            if not self._compose(host_filter, properties):
                return False
        except Exception as e:
            message = 'Could not evaluate host filter %s for host %s - %s' % (host_filter, name, to_native(e))
            if self.strict:
                raise AnsibleError(message)
            display.warning(message)
    return True