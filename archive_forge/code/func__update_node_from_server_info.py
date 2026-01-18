import json
import time
import datetime
from libcloud.utils.py3 import basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
def _update_node_from_server_info(self, node, server):
    node.id = server['id']
    node.name = server['name']
    if server['power'] == 'on':
        node.state = NodeState.RUNNING
    else:
        node.state = NodeState.STOPPED
    for network in server.get('networks', []):
        if network.get('network').startswith('wan-'):
            node.public_ips += network.get('ips', [])
        else:
            node.private_ips += network.get('ips', [])
    billing = server.get('billing', node.extra.get('billingcycle')).lower()
    if billing == self.EX_BILLINGCYCLE_HOURLY:
        node.extra['billingcycle'] = self.EX_BILLINGCYCLE_HOURLY
        node.extra['priceOn'] = server.get('priceHourlyOn')
        node.extra['priceOff'] = server.get('priceHourlyOff')
    else:
        node.extra['billingcycle'] = self.EX_BILLINGCYCLE_MONTHLY
        node.extra['priceOn'] = server.get('priceMonthlyOn')
        node.extra['priceOff'] = server.get('priceMonthlyOn')
    node.extra['location'] = self.ex_get_location(server['datacenter'])
    node.extra['dailybackup'] = server.get('backup') == '1'
    node.extra['managed'] = server.get('managed') == '1'
    return node