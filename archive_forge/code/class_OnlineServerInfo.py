from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.online import (
class OnlineServerInfo(Online):

    def __init__(self, module):
        super(OnlineServerInfo, self).__init__(module)
        self.name = 'api/v1/server'

    def _get_server_detail(self, server_path):
        try:
            return self.get(path=server_path).json
        except OnlineException as exc:
            self.module.fail_json(msg='A problem occurred while fetching: %s (%s)' % (server_path, exc))

    def all_detailed_servers(self):
        servers_api_path = self.get_resources()
        server_data = (self._get_server_detail(server_api_path) for server_api_path in servers_api_path)
        return [s for s in server_data if s is not None]