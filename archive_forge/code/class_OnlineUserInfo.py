from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.online import (
class OnlineUserInfo(Online):

    def __init__(self, module):
        super(OnlineUserInfo, self).__init__(module)
        self.name = 'api/v1/user'