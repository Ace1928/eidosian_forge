from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, ConfigLine
class VossNetworkConfig(NetworkConfig):

    def load(self, s):
        self._config_text = s
        self._items = voss_parse(s, self._indent)

    def _diff_line(self, other):
        updates = list()
        for item in self.items:
            if str(item) == 'exit':
                if updates and updates[-1]._parents:
                    updates.append(item)
            elif item not in other:
                updates.append(item)
        return updates