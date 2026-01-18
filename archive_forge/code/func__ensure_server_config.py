from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_server_config(self, server, server_params):
    """
        ensures the server is updated with the provided cpu and memory
        :param server: the CLC server object
        :param server_params: the dictionary of server parameters
        :return: (changed, group) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    cpu = server_params.get('cpu')
    memory = server_params.get('memory')
    changed = False
    result = None
    if not cpu:
        cpu = server.cpu
    if not memory:
        memory = server.memory
    if memory != server.memory or cpu != server.cpu:
        if not self.module.check_mode:
            result = self._modify_clc_server(self.clc, self.module, server.id, cpu, memory)
        changed = True
    return (changed, result)