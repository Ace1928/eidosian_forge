from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _restore_server_snapshot(self, server):
    """
        Restore snapshot for the CLC server
        :param server: the CLC server object
        :return: the restore snapshot request object from CLC API
        """
    result = None
    try:
        result = server.RestoreSnapshot()
    except CLCException as ex:
        self.module.fail_json(msg='Failed to restore snapshot for server : {0}. {1}'.format(server.id, ex.response_text))
    return result