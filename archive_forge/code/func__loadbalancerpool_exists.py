from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _loadbalancerpool_exists(self, alias, location, port, lb_id):
    """
        Checks to see if a pool exists on the specified port on the provided load balancer
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param port: the port to check and see if it exists
        :param lb_id: the id string of the provided load balancer
        :return: result: The id string of the pool or False
        """
    result = False
    try:
        pool_list = self.clc.v2.API.Call('GET', '/v2/sharedLoadBalancers/%s/%s/%s/pools' % (alias, location, lb_id))
    except APIFailedResponse as e:
        return self.module.fail_json(msg='Unable to fetch the load balancer pools for for load balancer id: {0}. {1}'.format(lb_id, str(e.response_text)))
    for pool in pool_list:
        if int(pool.get('port')) == int(port):
            result = pool.get('id')
    return result