import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestDescribeCluster(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<DescribeClusterResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <DescribeClusterResult>\n    <Cluster>\n      <Id>j-aaaaaaaaa</Id>\n      <Tags/>\n      <Ec2InstanceAttributes>\n        <Ec2AvailabilityZone>us-west-1c</Ec2AvailabilityZone>\n        <Ec2KeyName>my_secret_key</Ec2KeyName>\n      </Ec2InstanceAttributes>\n      <RunningAmiVersion>2.4.2</RunningAmiVersion>\n      <VisibleToAllUsers>true</VisibleToAllUsers>\n      <Status>\n        <StateChangeReason>\n          <Message>Terminated by user request</Message>\n          <Code>USER_REQUEST</Code>\n        </StateChangeReason>\n        <State>TERMINATED</State>\n        <Timeline>\n          <CreationDateTime>2014-01-24T01:21:21Z</CreationDateTime>\n          <ReadyDateTime>2014-01-24T01:25:26Z</ReadyDateTime>\n          <EndDateTime>2014-01-24T02:19:46Z</EndDateTime>\n        </Timeline>\n      </Status>\n      <AutoTerminate>false</AutoTerminate>\n      <Name>test analytics</Name>\n      <RequestedAmiVersion>2.4.2</RequestedAmiVersion>\n      <Applications>\n        <member>\n          <Name>hadoop</Name>\n          <Version>1.0.3</Version>\n        </member>\n      </Applications>\n      <TerminationProtected>false</TerminationProtected>\n      <MasterPublicDnsName>ec2-184-0-0-1.us-west-1.compute.amazonaws.com</MasterPublicDnsName>\n      <NormalizedInstanceHours>10</NormalizedInstanceHours>\n      <ServiceRole>my-service-role</ServiceRole>\n    </Cluster>\n  </DescribeClusterResult>\n  <ResponseMetadata>\n    <RequestId>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</RequestId>\n  </ResponseMetadata>\n</DescribeClusterResponse>\n        '

    def test_describe_cluster(self):
        self.set_http_response(200)
        with self.assertRaises(TypeError):
            self.service_connection.describe_cluster()
        response = self.service_connection.describe_cluster(cluster_id='j-123')
        self.assertTrue(isinstance(response, Cluster))
        self.assertEqual(response.id, 'j-aaaaaaaaa')
        self.assertEqual(response.runningamiversion, '2.4.2')
        self.assertEqual(response.visibletoallusers, 'true')
        self.assertEqual(response.autoterminate, 'false')
        self.assertEqual(response.name, 'test analytics')
        self.assertEqual(response.requestedamiversion, '2.4.2')
        self.assertEqual(response.terminationprotected, 'false')
        self.assertEqual(response.ec2instanceattributes.ec2availabilityzone, 'us-west-1c')
        self.assertEqual(response.ec2instanceattributes.ec2keyname, 'my_secret_key')
        self.assertEqual(response.status.state, 'TERMINATED')
        self.assertEqual(response.applications[0].name, 'hadoop')
        self.assertEqual(response.applications[0].version, '1.0.3')
        self.assertEqual(response.masterpublicdnsname, 'ec2-184-0-0-1.us-west-1.compute.amazonaws.com')
        self.assertEqual(response.normalizedinstancehours, '10')
        self.assertEqual(response.servicerole, 'my-service-role')
        self.assert_request_parameters({'Action': 'DescribeCluster', 'ClusterId': 'j-123', 'Version': '2009-03-31'})