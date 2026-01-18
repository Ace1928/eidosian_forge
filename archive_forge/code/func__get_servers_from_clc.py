from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_servers_from_clc(self, server_list, message):
    """
        Internal function to fetch list of CLC server objects from a list of server ids
        :param server_list: the list of server ids
        :param message: the error message to raise if there is any error
        :return the list of CLC server objects
        """
    try:
        return self.clc.v2.Servers(server_list).servers
    except CLCException as ex:
        self.module.fail_json(msg=message + ': %s' % ex)