import json
from tests.unit import AWSMockServiceTestCase
from boto.beanstalk.layer1 import Layer1
class TestListAvailableSolutionStacks(AWSMockServiceTestCase):
    connection_class = Layer1

    def default_body(self):
        return json.dumps({u'ListAvailableSolutionStacksResponse': {u'ListAvailableSolutionStacksResult': {u'SolutionStackDetails': [{u'PermittedFileTypes': [u'war', u'zip'], u'SolutionStackName': u'32bit Amazon Linux running Tomcat 7'}, {u'PermittedFileTypes': [u'zip'], u'SolutionStackName': u'32bit Amazon Linux running PHP 5.3'}], u'SolutionStacks': [u'32bit Amazon Linux running Tomcat 7', u'32bit Amazon Linux running PHP 5.3']}, u'ResponseMetadata': {u'RequestId': u'request_id'}}}).encode('utf-8')

    def test_list_available_solution_stacks(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.list_available_solution_stacks()
        stack_details = api_response['ListAvailableSolutionStacksResponse']['ListAvailableSolutionStacksResult']['SolutionStackDetails']
        solution_stacks = api_response['ListAvailableSolutionStacksResponse']['ListAvailableSolutionStacksResult']['SolutionStacks']
        self.assertEqual(solution_stacks, [u'32bit Amazon Linux running Tomcat 7', u'32bit Amazon Linux running PHP 5.3'])
        self.assert_request_parameters({'Action': 'ListAvailableSolutionStacks', 'ContentType': 'JSON', 'Version': '2010-12-01'})