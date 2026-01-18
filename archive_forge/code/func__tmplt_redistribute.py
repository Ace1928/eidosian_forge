from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_redistribute(redis):
    command = 'redistribute {protocol}'.format(**redis)
    if redis.get('id'):
        command += ' {id}'.format(**redis)
    if redis.get('metric'):
        command += ' metric {metric}'.format(**redis)
    if redis.get('level'):
        command += ' level {level}'.format(**redis)
    if redis.get('internal'):
        command += ' internal'
    if redis.get('external'):
        command += ' external'
    if redis.get('nssa_external'):
        command += ' nssa-external'
    if redis.get('external_ospf'):
        command += ' external {external_ospf}'.format(**redis)
    if redis.get('route_policy'):
        command += ' route-policy {route_policy}'.format(**redis)
    return command