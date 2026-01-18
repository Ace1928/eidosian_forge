from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _loadbalancerpool_nodes_exists(self, alias, location, lb_id, pool_id, nodes_to_check):
    """
        Checks to see if a set of nodes exists on the specified port on the provided load balancer
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the provided load balancer
        :param pool_id: the id string of the load balancer pool
        :param nodes_to_check: the list of nodes to check for
        :return: result: True / False indicating if the given nodes exist
        """
    result = False
    nodes = self._get_lbpool_nodes(alias, location, lb_id, pool_id)
    for node in nodes_to_check:
        if not node.get('status'):
            node['status'] = 'enabled'
        if node in nodes:
            result = True
        else:
            result = False
    return result