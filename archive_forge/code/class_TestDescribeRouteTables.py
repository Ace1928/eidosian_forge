from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestDescribeRouteTables(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeRouteTablesResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>6f570b0b-9c18-4b07-bdec-73740dcf861a</requestId>\n               <routeTableSet>\n                  <item>\n                     <routeTableId>rtb-13ad487a</routeTableId>\n                     <vpcId>vpc-11ad4878</vpcId>\n                     <routeSet>\n                        <item>\n                           <destinationCidrBlock>10.0.0.0/22</destinationCidrBlock>\n                           <gatewayId>local</gatewayId>\n                           <state>active</state>\n                           <origin>CreateRouteTable</origin>\n                        </item>\n                     </routeSet>\n                     <associationSet>\n                         <item>\n                            <routeTableAssociationId>rtbassoc-12ad487b</routeTableAssociationId>\n                            <routeTableId>rtb-13ad487a</routeTableId>\n                            <main>true</main>\n                         </item>\n                     </associationSet>\n                     <tagSet/>\n                  </item>\n                  <item>\n                     <routeTableId>rtb-f9ad4890</routeTableId>\n                     <vpcId>vpc-11ad4878</vpcId>\n                     <routeSet>\n                        <item>\n                           <destinationCidrBlock>10.0.0.0/22</destinationCidrBlock>\n                           <gatewayId>local</gatewayId>\n                           <state>active</state>\n                           <origin>CreateRouteTable</origin>\n                        </item>\n                        <item>\n                           <destinationCidrBlock>0.0.0.0/0</destinationCidrBlock>\n                           <gatewayId>igw-eaad4883</gatewayId>\n                           <state>active</state>\n                            <origin>CreateRoute</origin>\n                        </item>\n                        <item>\n                            <destinationCidrBlock>10.0.0.0/21</destinationCidrBlock>\n                            <networkInterfaceId>eni-884ec1d1</networkInterfaceId>\n                            <state>blackhole</state>\n                            <origin>CreateRoute</origin>\n                        </item>\n                        <item>\n                            <destinationCidrBlock>11.0.0.0/22</destinationCidrBlock>\n                            <vpcPeeringConnectionId>pcx-efc52b86</vpcPeeringConnectionId>\n                            <state>blackhole</state>\n                            <origin>CreateRoute</origin>\n                        </item>\n                     </routeSet>\n                     <associationSet>\n                        <item>\n                            <routeTableAssociationId>rtbassoc-faad4893</routeTableAssociationId>\n                            <routeTableId>rtb-f9ad4890</routeTableId>\n                            <subnetId>subnet-15ad487c</subnetId>\n                        </item>\n                     </associationSet>\n                     <tagSet/>\n                  </item>\n               </routeTableSet>\n            </DescribeRouteTablesResponse>\n        '

    def test_get_all_route_tables(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_route_tables(['rtb-13ad487a', 'rtb-f9ad4890'], filters=[('route.state', 'active')])
        self.assert_request_parameters({'Action': 'DescribeRouteTables', 'RouteTableId.1': 'rtb-13ad487a', 'RouteTableId.2': 'rtb-f9ad4890', 'Filter.1.Name': 'route.state', 'Filter.1.Value.1': 'active'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(len(api_response), 2)
        self.assertIsInstance(api_response[0], RouteTable)
        self.assertEquals(api_response[0].id, 'rtb-13ad487a')
        self.assertEquals(len(api_response[0].routes), 1)
        self.assertEquals(api_response[0].routes[0].destination_cidr_block, '10.0.0.0/22')
        self.assertEquals(api_response[0].routes[0].gateway_id, 'local')
        self.assertEquals(api_response[0].routes[0].state, 'active')
        self.assertEquals(api_response[0].routes[0].origin, 'CreateRouteTable')
        self.assertEquals(len(api_response[0].associations), 1)
        self.assertEquals(api_response[0].associations[0].id, 'rtbassoc-12ad487b')
        self.assertEquals(api_response[0].associations[0].route_table_id, 'rtb-13ad487a')
        self.assertIsNone(api_response[0].associations[0].subnet_id)
        self.assertEquals(api_response[0].associations[0].main, True)
        self.assertEquals(api_response[1].id, 'rtb-f9ad4890')
        self.assertEquals(len(api_response[1].routes), 4)
        self.assertEquals(api_response[1].routes[0].destination_cidr_block, '10.0.0.0/22')
        self.assertEquals(api_response[1].routes[0].gateway_id, 'local')
        self.assertEquals(api_response[1].routes[0].state, 'active')
        self.assertEquals(api_response[1].routes[0].origin, 'CreateRouteTable')
        self.assertEquals(api_response[1].routes[1].destination_cidr_block, '0.0.0.0/0')
        self.assertEquals(api_response[1].routes[1].gateway_id, 'igw-eaad4883')
        self.assertEquals(api_response[1].routes[1].state, 'active')
        self.assertEquals(api_response[1].routes[1].origin, 'CreateRoute')
        self.assertEquals(api_response[1].routes[2].destination_cidr_block, '10.0.0.0/21')
        self.assertEquals(api_response[1].routes[2].interface_id, 'eni-884ec1d1')
        self.assertEquals(api_response[1].routes[2].state, 'blackhole')
        self.assertEquals(api_response[1].routes[2].origin, 'CreateRoute')
        self.assertEquals(api_response[1].routes[3].destination_cidr_block, '11.0.0.0/22')
        self.assertEquals(api_response[1].routes[3].vpc_peering_connection_id, 'pcx-efc52b86')
        self.assertEquals(api_response[1].routes[3].state, 'blackhole')
        self.assertEquals(api_response[1].routes[3].origin, 'CreateRoute')
        self.assertEquals(len(api_response[1].associations), 1)
        self.assertEquals(api_response[1].associations[0].id, 'rtbassoc-faad4893')
        self.assertEquals(api_response[1].associations[0].route_table_id, 'rtb-f9ad4890')
        self.assertEquals(api_response[1].associations[0].subnet_id, 'subnet-15ad487c')
        self.assertEquals(api_response[1].associations[0].main, False)