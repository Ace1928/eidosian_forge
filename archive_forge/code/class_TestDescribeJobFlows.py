import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestDescribeJobFlows(DescribeJobFlowsTestBase):

    def test_describe_jobflows_response(self):
        self.set_http_response(200)
        response = self.service_connection.describe_jobflows()
        self.assertTrue(isinstance(response, list))
        jf = response[0]
        self.assertTrue(isinstance(jf, JobFlow))
        self.assertEqual(jf.amiversion, '2.4.2')
        self.assertEqual(jf.visibletoallusers, 'true')
        self.assertEqual(jf.name, 'test analytics')
        self.assertEqual(jf.jobflowid, 'j-aaaaaa')
        self.assertEqual(jf.ec2keyname, 'my_key')
        self.assertEqual(jf.masterinstancetype, 'm1.large')
        self.assertEqual(jf.availabilityzone, 'us-west-1c')
        self.assertEqual(jf.keepjobflowalivewhennosteps, 'true')
        self.assertEqual(jf.slaveinstancetype, 'm1.large')
        self.assertEqual(jf.masterinstanceid, 'i-aaaaaa')
        self.assertEqual(jf.hadoopversion, '1.0.3')
        self.assertEqual(jf.normalizedinstancehours, '12')
        self.assertEqual(jf.masterpublicdnsname, 'ec2-184-0-0-1.us-west-1.compute.amazonaws.com')
        self.assertEqual(jf.instancecount, '3')
        self.assertEqual(jf.terminationprotected, 'false')
        self.assertTrue(isinstance(jf.steps, list))
        step = jf.steps[0]
        self.assertTrue(isinstance(step, Step))
        self.assertEqual(step.jar, 's3://us-west-1.elasticmapreduce/libs/script-runner/script-runner.jar')
        self.assertEqual(step.name, 'Setup hive')
        self.assertEqual(step.actiononfailure, 'TERMINATE_JOB_FLOW')
        self.assertTrue(isinstance(jf.instancegroups, list))
        ig = jf.instancegroups[0]
        self.assertTrue(isinstance(ig, InstanceGroup))
        self.assertEqual(ig.creationdatetime, '2014-01-24T01:21:21Z')
        self.assertEqual(ig.state, 'ENDED')
        self.assertEqual(ig.instancerequestcount, '1')
        self.assertEqual(ig.instancetype, 'm1.large')
        self.assertEqual(ig.laststatechangereason, 'Job flow terminated')
        self.assertEqual(ig.market, 'ON_DEMAND')
        self.assertEqual(ig.instancegroupid, 'ig-aaaaaa')
        self.assertEqual(ig.instancerole, 'MASTER')
        self.assertEqual(ig.name, 'Master instance group')

    def test_describe_jobflows_no_args(self):
        self.set_http_response(200)
        self.service_connection.describe_jobflows()
        self.assert_request_parameters({'Action': 'DescribeJobFlows'}, ignore_params_values=['Version'])

    def test_describe_jobflows_filtered(self):
        self.set_http_response(200)
        now = datetime.now()
        a_bit_before = datetime.fromtimestamp(time() - 1000)
        self.service_connection.describe_jobflows(states=['WAITING', 'RUNNING'], jobflow_ids=['j-aaaaaa', 'j-aaaaab'], created_after=a_bit_before, created_before=now)
        self.assert_request_parameters({'Action': 'DescribeJobFlows', 'JobFlowIds.member.1': 'j-aaaaaa', 'JobFlowIds.member.2': 'j-aaaaab', 'JobFlowStates.member.1': 'WAITING', 'JobFlowStates.member.2': 'RUNNING', 'CreatedAfter': a_bit_before.strftime(boto.utils.ISO8601), 'CreatedBefore': now.strftime(boto.utils.ISO8601)}, ignore_params_values=['Version'])