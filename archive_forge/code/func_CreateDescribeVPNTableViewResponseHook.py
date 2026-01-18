from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CreateDescribeVPNTableViewResponseHook(response, args):
    """Create DescribeVPNTableView from GetVpnConnection response.

  Args:
    response: Response from GetVpnConnection
    args: Args from GetVpnConnection

  Returns:
    DescribeVPNTableView
  """
    del args
    name = response.name
    create_time = response.createTime
    details = response.details
    if details:
        state = details.state
        error = details.error
    else:
        state = 'STATE_UNKNOWN'
        error = ''
    cluster = {}
    items = response.cluster.split('/')
    try:
        cluster['project'] = items[1]
        cluster['location'] = items[3]
        cluster['ID'] = items[5]
    except IndexError:
        pass
    if response.natGatewayIp:
        cluster['NAT Gateway IP'] = response.natGatewayIp
    vpc = {}
    items = response.vpc.split('/')
    try:
        vpc['project'] = items[1]
        vpc['ID'] = items[5]
    except IndexError:
        pass
    if details:
        vpc['Cloud Router'] = {'name': details.cloudRouter.name, 'region': items[3]}
        vpc['Cloud VPNs'] = details.cloudVpns
    return DescribeVPNTableView(name, create_time, cluster, vpc, state, error)