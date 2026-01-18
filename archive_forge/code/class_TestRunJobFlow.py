import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestRunJobFlow(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<RunJobFlowResponse xmlns="http://elasticmapreduce.amazonaws.com/doc/2009-03-31">\n  <RunJobFlowResult>\n    <JobFlowId>j-ZKIY4CKQRX72</JobFlowId>\n  </RunJobFlowResult>\n  <ResponseMetadata>\n    <RequestId>aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee</RequestId>\n  </ResponseMetadata>\n</RunJobFlowResponse>\n'

    def test_run_jobflow_service_role(self):
        self.set_http_response(200)
        response = self.service_connection.run_jobflow('EmrCluster', service_role='EMR_DefaultRole')
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'RunJobFlow', 'Version': '2009-03-31', 'ServiceRole': 'EMR_DefaultRole', 'Name': 'EmrCluster'}, ignore_params_values=['ActionOnFailure', 'Instances.InstanceCount', 'Instances.KeepJobFlowAliveWhenNoSteps', 'Instances.MasterInstanceType', 'Instances.SlaveInstanceType'])

    def test_run_jobflow_enable_debugging(self):
        self.region = 'ap-northeast-2'
        self.set_http_response(200)
        self.service_connection.run_jobflow('EmrCluster', enable_debugging=True)
        actual_params = set(self.actual_request.params.copy().items())
        expected_params = set([('Steps.member.1.HadoopJarStep.Jar', 's3://ap-northeast-2.elasticmapreduce/libs/script-runner/script-runner.jar'), ('Steps.member.1.HadoopJarStep.Args.member.1', 's3://ap-northeast-2.elasticmapreduce/libs/state-pusher/0.1/fetch')])
        self.assertTrue(expected_params <= actual_params)