import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
class TestAddJobFlowSteps(AWSMockServiceTestCase):
    connection_class = EmrConnection

    def default_body(self):
        return b'\n<AddJobFlowStepsOutput>\n    <StepIds>\n        <member>Foo</member>\n        <member>Bar</member>\n    </StepIds>\n</AddJobFlowStepsOutput>\n'

    def test_add_jobflow_steps(self):
        self.set_http_response(200)
        response = self.service_connection.add_jobflow_steps(jobflow_id='j-123', steps=[])
        self.assertTrue(isinstance(response, JobFlowStepList))
        self.assertEqual(response.stepids[0].value, 'Foo')
        self.assertEqual(response.stepids[1].value, 'Bar')