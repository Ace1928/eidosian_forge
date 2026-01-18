from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _remove_publicip_from_server(self, server):
    result = None
    try:
        for ip_address in server.PublicIPs().public_ips:
            result = ip_address.Delete()
    except CLCException as ex:
        self.module.fail_json(msg='Failed to remove public ip from the server : {0}. {1}'.format(server.id, ex.response_text))
    return result