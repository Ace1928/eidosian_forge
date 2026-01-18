from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.online import (
def _get_server_detail(self, server_path):
    try:
        return self.get(path=server_path).json
    except OnlineException as exc:
        self.module.fail_json(msg='A problem occurred while fetching: %s (%s)' % (server_path, exc))