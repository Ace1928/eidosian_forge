from boto.ec2.connection import EC2Connection
from boto.resultset import ResultSet
from boto.vpc.vpc import VPC
from boto.vpc.customergateway import CustomerGateway
from boto.vpc.networkacl import NetworkAcl
from boto.vpc.routetable import RouteTable
from boto.vpc.internetgateway import InternetGateway
from boto.vpc.vpngateway import VpnGateway, Attachment
from boto.vpc.dhcpoptions import DhcpOptions
from boto.vpc.subnet import Subnet
from boto.vpc.vpnconnection import VpnConnection
from boto.vpc.vpc_peering_connection import VpcPeeringConnection
from boto.ec2 import RegionData
from boto.regioninfo import RegionInfo, get_regions
from boto.regioninfo import connect
def accept_vpc_peering_connection(self, vpc_peering_connection_id, dry_run=False):
    """
        Acceptss a VPC peering connection request. The VPC peering connection 
        must be in the pending-acceptance state. 

        :type vpc_peering_connection_id: str
        :param vpc_peering_connection_id: The ID of the VPC peering connection.

        :rtype: Accepted VpcPeeringConnection
        :return: A :class:`boto.vpc.vpc_peering_connection.VpcPeeringConnection` object
        """
    params = {'VpcPeeringConnectionId': vpc_peering_connection_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('AcceptVpcPeeringConnection', params, VpcPeeringConnection)